#!/usr/bin/env python
"""B2: E4 tren nen KTN-R (E4R).
Khong sua b1gop.py, ktnr_buoc1.py; dung lai ham cua b1gop.py, b1moi_lai.py.
Nhanh R: cac ham trca, dau_max, w_tu, co_so, diem_R, z la BAN SAO NGUYEN VAN cua ktnr_buoc1.py) vi ktnr_buoc1.py
nap them bo du lieu khong co tren maychu4; dung ban sao duoc kiem bang cong hoi quy z_R, z_ktn tung luot so voi JSON KTN-R.
Hang so khoa: ALPHA = 0,5; H = 5; K = 5; N0 = 40; eta nhu E4 (he so 1,0).
Cac o (cham luot 41..het):
  KTN  : quyet dinh cua logit KTN tinh nhu ktnr_buoc1 (me LO = 20 luot tu N0).
  KTNR : argmax[z(logit KTN) + ALPHA z(R)].
  E4   : b1gop.phat_lai_eg (ban sao, them ghi logit cua tung luot TRUOC cap nhat).
  E4R-a: lam cap nhat dung nhu E4 (mat mat -do_tin(logit)); quyet dinh = argmax[z(logit(lam_t)) + ALPHA z(R_t)].
  E4R-b: mat mat -do_tin(s), s = z(logit(lam_t)) + ALPHA z(R_t) (z(R_t) hang so theo lam); quyet dinh = argmax s.
Quyet dinh a, b tinh bang numpy float64 nhu KTN-R (z: TB, sd tong the qua lop, + 1e-12); gradient cua b tinh bang torch tren cung dinh nghia.
Cong hoi quy (moi fold, lech thi dung): nguon chon trung JSON b1gop va KTN-R; E0 trung JSON b1gop tung quyet dinh; E4 trung JSON b1gop tung
quyet dinh; z_ktn, z_R lech JSON KTN-R toi da 1e-5 (JSON lam tron 6 chu so); KTNR tinh lai trung do chinh xac JSON KTN-R.
--kiem: bao them so quyet dinh khac giua E4 va E4R-b voi ALPHA = 0 (ly thuyet: trung, vi do_tin bat bien voi bien doi affine duong; chi bao, khong dung).
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch
import scipy.linalg as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1gop as G            # noqa: E402  (dat root du lieu theo bien moi truong B1_ROOT_*)
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402
import b1moi_lai as L        # noqa: E402
import du_lieu_lee as DL     # noqa: E402

SEED, N0, NCH, K5 = G.SEED, G.N0, G.NCH, G.K_CHINH
ALPHA, H = 0.5, 5
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---- ban sao nguyen van tu ktnr_buoc1.py ----
def trca(U, y, NT):
    """U (n, C, ns) da dinh tam -> w (C,), |w| = 1."""
    C = U.shape[1]
    Q = np.einsum("ict,idt->cd", U, U)
    S = -Q.copy()
    for c in range(NT):
        Uc = U[y == c].sum(0)
        S += Uc @ Uc.T
    Qr = Q + 1e-6 * np.trace(Q) / C * np.eye(C)
    lam, V = sl.eigh(S, Qr)
    w = V[:, dau_max(lam)]
    return w / np.linalg.norm(w)


def dau_max(lam):
    """Chi so DAU TIEN trong nhom tri rieng bang tri lon nhat."""
    m = lam.max()
    return int(np.where(lam >= m - 1e-12 * max(1.0, abs(m)))[0][0])


def w_tu(ws):
    M = sum(np.outer(w, w) for w in ws) / len(ws)
    lam, V = np.linalg.eigh(M)
    return V[:, dau_max(lam)]


def co_so(freqs, fs, ns):
    t = np.arange(ns) / fs
    out = []
    for f in freqs:
        Y = np.stack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)], 1)   # (ns, 2H)
        out.append(np.linalg.qr(Y)[0])
    return np.stack(out)                                                                                  # (N, ns, 2H)


def diem_R(yv, Qb):
    """yv (n, ns) da tru TB; Qb (N, ns, 2H) -> (n, N)."""
    p = np.einsum("it,ntk->ink", yv, Qb)
    ny = np.linalg.norm(yv, axis=1)
    R = np.sqrt((p ** 2).sum(-1)) / np.where(ny > 0, ny, 1.0)[:, None]
    R[ny == 0] = 0.0
    return R


def z(v):
    return (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-12)
# ---- het ban sao ----


def z_t(v):
    """Cung dinh nghia z cho tensor (1, N), de lay gradient (E4R-b)."""
    return (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True, unbiased=False) + 1e-12)


def phat_lai(model, h1, lg_c, DW, DB, zR, alpha, day_gop):
    """Gradient mu tung luot nhu b1gop.phat_lai_eg (he so 1,0). day_gop = False: mat mat -do_tin(logit) (E4, E4R-a).
    day_gop = True: mat mat -do_tin(z(logit) + alpha z(R_t)) (E4R-b). Tra: dec_logit (n,), logit truoc cap nhat cua luot N0.. (T, N), them."""
    K = DW.shape[0]; n = h1.shape[0]; dev = h1.device; T = n - N0
    dec = torch.empty(n, dtype=torch.long, device=dev); dec[:N0] = lg_c[:N0].argmax(1)
    theta = torch.zeros(K, device=dev); Gm = 0.0; eta = 0.0; bo_khong_huu_han = 0; bo_G0 = 0
    lgs = []
    zRt = torch.tensor(zR, device=dev, dtype=torch.float32)
    for t in range(N0, n):
        lv = torch.softmax(theta, 0).requires_grad_(True)
        lg = L.sau(model, h1[t:t + 1], G.gop(DW, DB, lv))
        dec[t] = lg.argmax(1)[0]
        lgs.append(lg.detach().double().cpu().numpy()[0])
        dt = L.do_tin(z_t(lg) + alpha * zRt[t - N0:t - N0 + 1]) if day_gop else L.do_tin(lg)
        (g,) = torch.autograd.grad(-dt.sum(), lv)
        if not torch.isfinite(g).all():
            bo_khong_huu_han += 1; continue
        Gm = max(Gm, float(g.abs().max()))
        if Gm > 0:
            eta = math.sqrt(2.0 * math.log(K) / (Gm * Gm * T))
            theta = theta - eta * g
        else:
            bo_G0 += 1
    lam = torch.softmax(theta, 0).cpu().numpy()
    them = dict(lam=lam.tolist(), tv=G.tv(lam), G=Gm, eta_cuoi=eta, bo_khong_huu_han=bo_khong_huu_han, bo_G0=bo_G0)
    return dec.cpu().numpy(), np.array(lgs), them


def chay_fold(i, D, a, dev, freqs, fs, NT, gg, gk, kiem):
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))                  # nhu b1gop.chay_fold va ktnr_buoc1.chay_fold
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    assert gg["seed"] == seed and gk["seed"] == seed
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95).to(dev)
    ck = os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")
    assert sha(ck) == gk["trongso_sha256"], f"trong so fold {i+1} khac JSON KTN-R"
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev); y = D["y"][i]; yt = y[N0:]
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():
        h1u = L.truoc(model, Xu); lg0 = model(Xu)
        lg_c = L.sau(model, h1u, mc)
        assert torch.allclose(lg0, lg_c, atol=1e-4)
    mods, wsrc = [], []
    for s in src:                                                        # nhu b1gop va ktnr_buoc1 (cung thu tu goi)
        Xs = torch.tensor(D["X"][s], device=dev); ys = torch.tensor(D["y"][s], device=dev, dtype=torch.long)
        with torch.no_grad():
            h1s = L.truoc(model, Xs)
        mods.append(L.hoc(model, h1s, ys, mc, L.BUOC_NGUON, L.LR_MOD))
        Us = h1s[:, 0].double().cpu().numpy(); Us = Us - Us.mean(-1, keepdims=True)
        wsrc.append(trca(Us, D["y"][s], NT))
    with torch.no_grad():
        lgs = torch.stack([L.sau(model, h1u, m) for m in mods])
    g = L.do_tin(lgs[:, :N0]).mean(1)
    top = torch.argsort(-g, stable=True)[:K5]
    chon = [src[int(j)] + 1 for j in top]
    assert chon == gg["nguon_chon5"] and chon == gk["nguon_chon"], f"nguon chon lech o fold {i+1}"
    DW5, DB5 = G.xep(mods, top)
    ms = {k: torch.stack([mods[int(j)][k] for j in top]).mean(0) for k in mc}
    e0 = G.phat_lai_co_dinh(model, h1u, lg_c, ms)
    assert gg["E0"]["dec"] == [int(v) for v in e0], f"hoi quy E0 lech o fold {i+1}"
    lk_t = []
    with torch.no_grad():                                                # logit KTN nhu ktnr_buoc1 (me LO tu N0)
        for a_ in range(N0, h1u.shape[0], L.LO):
            lk_t.append(L.sau(model, h1u[a_:a_ + L.LO], {k: v.clone() for k, v in ms.items()}))
    lk = torch.cat(lk_t).double().cpu().numpy()
    U = h1u[N0:, 0].double().cpu().numpy(); U = U - U.mean(-1, keepdims=True)
    R5 = diem_R(np.einsum("c,ict->it", w_tu([wsrc[int(j)] for j in top]), U), co_so(freqs, fs, ns))
    zk, zR = z(lk), z(R5)
    assert np.isfinite(zk).all() and np.isfinite(zR).all()
    lch_k = float(np.abs(zk - np.array(gk["luot"]["z_ktn"])).max()); lch_R = float(np.abs(zR - np.array(gk["luot"]["z_R"])).max())
    assert lch_k < 1e-5 and lch_R < 1e-5, f"hoi quy z lech o fold {i+1}: z_ktn {lch_k:.2e}, z_R {lch_R:.2e}"
    d_ktn = zk.argmax(1); d_ktnr = (zk + ALPHA * zR).argmax(1)
    assert float((d_ktnr == yt).mean()) == gk["ktnr"], f"hoi quy KTNR lech o fold {i+1}"
    # kiem anh xa luot va tung quyet dinh KTNR so voi tham chieu da lam tron (chung nhan argmax theo bien sai so)
    assert gk["luot"]["y"] == [int(v) for v in yt] and gk["luot"]["chi_so"] == list(range(N0 + 1, N0 + 1 + len(yt))), f"anh xa luot lech o fold {i+1}"
    ref = np.array(gk["luot"]["z_ktn"]) + ALPHA * np.array(gk["luot"]["z_R"])
    assert ref.shape == zk.shape == (len(yt), NT)
    tol = 2 * (lch_k + ALPHA * lch_R) + 1e-12                             # chenh hai lop cua diem tinh lai so voi tham chieu bi chan boi tol
    srt = np.sort(ref, 1); bien = srt[:, -1] - srt[:, -2]
    chung_nhan = bien > tol
    assert np.array_equal(d_ktnr[chung_nhan], ref.argmax(1)[chung_nhan]), f"quyet dinh KTNR lech tham chieu o fold {i+1}"
    gan = ref >= (srt[:, -1:] - tol)                                      # luot khong chung nhan: quyet dinh phai thuoc nhom lop sat dau cua tham chieu
    assert all(gan[k_, d_ktnr[k_]] for k_ in np.where(~chung_nhan)[0]), f"quyet dinh KTNR ngoai nhom sat dau o fold {i+1}"
    so_khong_chung_nhan = int((~chung_nhan).sum())
    e4, lg4, th4 = phat_lai(model, h1u, lg_c, DW5, DB5, zR, ALPHA, False)
    assert gg["E4"]["dec"] == [int(v) for v in e4], f"hoi quy E4 lech o fold {i+1}"
    eb, lgb, thb = phat_lai(model, h1u, lg_c, DW5, DB5, zR, ALPHA, True)
    assert np.isfinite(lg4).all() and np.isfinite(lgb).all() and np.isfinite(thb["lam"]).all(), f"logit hoac lam khong huu han o fold {i+1}"
    d_a = (z(lg4) + ALPHA * zR).argmax(1)
    d_b = (z(lgb) + ALPHA * zR).argmax(1)
    acc = lambda d: float((d == yt).mean())
    res = dict(fold=i + 1, seed=seed, nguon_chon5=chon, lech_z_ktn=lch_k, lech_z_R=lch_R, so_luot_ktnr_khong_chung_nhan=so_khong_chung_nhan,
               n_cham=int(len(yt)), bo_khong_huu_han_E4=th4["bo_khong_huu_han"], bo_G0_E4=th4["bo_G0"],
               KTN=acc(d_ktn), KTNR=acc(d_ktnr), E0=acc(e0[N0:]), E4=acc(e4[N0:]), E4Ra=acc(d_a), E4Rb=acc(d_b),
               P4_a=float((d_a != d_ktnr).mean()), P4_b=float((d_b != d_ktnr).mean()), P4_E4=float((e4[N0:] != e0[N0:]).mean()),
               E4Rb_logit_rieng=acc(eb[N0:]),
               lam_E4=th4["lam"], tv_E4=th4["tv"], G_E4=th4["G"], lam_b=thb["lam"], tv_b=thb["tv"], G_b=thb["G"], eta_cuoi_b=thb["eta_cuoi"],
               bo_khong_huu_han_b=thb["bo_khong_huu_han"], bo_G0_b=thb["bo_G0"],
               dec=dict(KTNR=[int(v) for v in d_ktnr], E4Ra=[int(v) for v in d_a], E4Rb=[int(v) for v in d_b]))
    if kiem:
        e4k, _, _ = phat_lai(model, h1u, lg_c, DW5, DB5, zR, 0.0, True)
        res["kiem_alpha0_so_quyet_dinh_khac_E4"] = int((e4k != e4).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta"])
    ap.add_argument("--T", type=float, required=True, choices=[0.6, 1.0])
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--json-goc", required=True, help="JSON b1gop buoc 1 (benchmark) hoac 2a (beta) cung T")
    ap.add_argument("--json-ktnr", required=True, help="JSON KTN-R buoc 1 cung bo, cung T")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all", help="vi du 1-2")
    ap.add_argument("--kiem", action="store_true")
    ap.add_argument("--thiet-bi", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    if a.thiet_bi == "cuda" and not torch.cuda.is_available():
        sys.exit("khong co CUDA")
    dev = torch.device(a.thiet_bi)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    ds = a.ds; DL.dang_ky(B, GD, ds); assert GD.NT == DL.SO_LOP[ds]; nl = DL.SO_LUOT[ds]; c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    gg = json.load(open(a.json_goc)); gk = json.load(open(a.json_ktnr))
    assert gg["ds"] == ds and abs(gg["T"] - a.T) < 1e-9 and gg["trongso"] == a.trongso, "JSON b1gop khac bo, cua so hoac trong so"
    assert gk["ds"] == ds and abs(gk["T"] - a.T) < 1e-9 and gk["alpha"] == ALPHA and gk["H"] == H and gk["k"] == K5 and gk["n0"] == N0, "JSON KTN-R khac cau hinh"
    assert [p["fold"] for p in gk["tung_nguoi"]] == list(range(1, nsub + 1)) and [p["fold"] for p in gg["tung_nguoi"]] == list(range(1, nsub + 1))
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))
    assert cfg["nch"] == NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg["subjects"] == 0 and cfg.get("ds", "benchmark") == ds
    assert cfg["script_sha256"] == gg["trongso_script_sha256"] == gk["trongso_script_sha256"]
    folds = list(range(nsub)) if a.folds == "all" else list(range(int(a.folds.split("-")[0]) - 1, int(a.folds.split("-")[1])))
    D = {"X": [], "y": []}; freqs = None
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and X.shape[1] == 9 and np.isfinite(X).all() and np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, nl // GD.NT))
        freqs = np.asarray(fr, float) if freqs is None else freqs
        assert np.allclose(freqs, fr)
        D["X"].append(GD.tien_xu_ly(X, ds, "causal", ns, fs)); D["y"].append(y)
    assert np.allclose(freqs, gk["freqs"]) and ns == gk["ns"] and fs == gk["fs"]
    print("thiet bi:", dev, "| numpy", np.__version__, "| torch", torch.__version__, flush=True)
    t0 = time.time(); per = []
    for i in folds:
        per.append(chay_fold(i, D, a, dev, freqs, fs, GD.NT, gg["tung_nguoi"][i], gk["tung_nguoi"][i], a.kiem))
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"fold {i+1} chay thong {time.time()-t0:.0f} s", flush=True)      # khong in do chinh xac trong luc chay
    js = dict(seed=SEED, ds=ds, T=a.T, n0=N0, alpha=ALPHA, H=H, k=K5, thiet_bi=str(dev), numpy=np.__version__, torch=torch.__version__,
              script_sha256=sha(os.path.abspath(__file__)), b1gop_sha256=sha(G.__file__),
              b1moi_lai_sha256=sha(L.__file__), global_dnn_sha256=sha(GD.__file__), b1_diem_sha256=sha(B.__file__),
              json_goc=a.json_goc, json_goc_sha256=sha(a.json_goc), json_ktnr=a.json_ktnr, json_ktnr_sha256=sha(a.json_ktnr),
              ktnr_script_sha256=gk["script_sha256"], trongso=a.trongso,
              trongso_fold_sha256=[sha(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")) for i in folds],
              du_fold=(len(folds) == nsub), tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"))
    os.link(a.out + ".tmp", a.out); os.remove(a.out + ".tmp")          # os.link that bai neu tep ket qua da co (khong ghi de)
    if os.path.exists(a.out + ".dang-chay.tmp"):
        os.remove(a.out + ".dang-chay.tmp")
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
