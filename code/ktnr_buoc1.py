#!/usr/bin/env python
"""KTN-R buoc 1.
Phat lai KTN (dung ham cua b1moi_lai.py, khong sua tep do) tren Benchmark / BETA / eldBETA bang mang nen va mo-dun nguon nhu job 18691/18692/18700,
roi them nhanh R:
  w_s  : TRCA gop lop tren u = l1(x) (dinh tam theo thoi gian moi luot) cua nguoi nguon s (co nhan); S = sum_lop (U_c U_c^T - sum_i X_i X_i^T),
         Q = sum_i X_i X_i^T; giai S w = lam (Q + 1e-6 tr(Q)/C I) w; w = vecto rieng tri rieng lon nhat, |w| = 1.
  w_R  : vecto rieng lon nhat cua (1/5) sum w_s w_s^T tren DUNG 5 nguon KTN chon (khong can dau).
  R_c  : y = w_R^T u (tru TB theo thoi gian); Q_c = QR cua [sin, cos] hai 1..H cua f_c (KHONG dinh tam); R_c = |Q_c^T y| / |y|; |y| = 0 -> 0.
  gop  : z(v) = (v - TB lop)/(sd tong the qua lop + 1e-12); s = z(logit KTN) + ALPHA z(R); argmax, hoa lay lop nho nhat (torch.argmax tra chi so dau).
Hang so khoa: ALPHA = 0,5; H = 5; K = 5 nguon; n0 = 40.
Bien the (cung ALPHA, H, z; cham luot 41..het): ktn, ktnr; doi chung D-R: ktnr_ngau5 (w_R tu 5 nguon ngau nhien, 20 lan, rng(seed + 992), TB),
ktnr_tatca (w_R tu moi nguon), ktnr_oz (y = kenh OZ cua u, khong loc khong gian), tinh_r (mang tinh + R tu 5 nguon chon), r_rieng (argmax R);
bao kem luoi alpha {0,25; 1} (KHONG doi alpha); rho = Pearson giua le z cua lop dung (z lop dung - TB z lop sai) cua logit KTN va R, luot 41..het.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
import torch
import scipy.linalg as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402
import du_lieu_eld as DE     # noqa: E402
import b1moi_lai as L        # noqa: E402
import du_lieu_nak as DN     # noqa: E402  (them 2026-11-16: buoc 2, Nakanishi; ba bo cu khong doi phep tinh)
import du_lieu_dong as DG    # noqa: E402  (them 2026-11-17: xac nhan Dong 2023)

SEED = int(os.environ.get("B2_SEED", "20261107"))   # seed huan luyen va phat lai; mac dinh nhu bai
ALPHA, H, K, N0, R_DR = 0.5, 5, 5, 40, 20
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


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


def le(zv, y):
    n, N = zv.shape
    dung = zv[np.arange(n), y]
    return dung - (zv.sum(1) - dung) / (N - 1)


def chay_fold(i, D, a, dev, freqs, fs, NT):
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=L.NCH, p_cuoi=0.95).to(dev)
    ck = os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev); y = D["y"][i]
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():
        h1u = L.truoc(model, Xu); lg0 = model(Xu)
    mods, wsrc = [], []
    for s in src:                                                        # giong het vong mo-dun nguon cua b1moi_lai.chay_fold (kieu B, cung thu tu goi RNG)
        Xs = torch.tensor(D["X"][s], device=dev); ys = torch.tensor(D["y"][s], device=dev, dtype=torch.long)
        with torch.no_grad():
            h1s = L.truoc(model, Xs)
        mods.append(L.hoc(model, h1s, ys, mc, L.BUOC_NGUON, L.LR_MOD))
        Us = h1s[:, 0].double().cpu().numpy(); Us = Us - Us.mean(-1, keepdims=True)
        wsrc.append(trca(Us, D["y"][s], NT))
    with torch.no_grad():
        lgs = torch.stack([L.sau(model, h1u, m) for m in mods])
    if not torch.isfinite(lgs).all():
        raise SystemExit(f"fold {i+1}: logit nguon khong huu han")
    g = L.do_tin(lgs[:, :N0]).mean(1)
    top = torch.argsort(-g, stable=True)[:K]
    ms = {k: torch.stack([mods[int(j)][k] for j in top]).mean(0) for k in mc}
    n_ = h1u.shape[0]; lk_t = []
    with torch.no_grad():                                                # dung nhu b1moi_lai.phat_lai_mod: me LO = 20 luot tu N0, argmax bang torch
        for a_ in range(N0, n_, L.LO):
            lk_t.append(L.sau(model, h1u[a_:a_ + L.LO], {k: v.clone() for k, v in ms.items()}))
    lk_t = torch.cat(lk_t)
    ktn_dec = lk_t.argmax(1).cpu().numpy()
    lk = lk_t.double().cpu().numpy()
    l0 = lg0[N0:].double().cpu().numpy()
    yt = y[N0:]
    U = h1u[N0:, 0].double().cpu().numpy(); U = U - U.mean(-1, keepdims=True)                   # (n, C, ns)
    Qb = co_so(freqs, fs, ns)
    Rof = lambda w: diem_R(np.einsum("c,ict->it", w, U) - 0.0, Qb)                               # U da dinh tam nen y da tru TB
    hit = lambda s_: float((s_.argmax(1) == yt).mean())
    tch = [int(j) for j in top.cpu().numpy()]
    R5 = Rof(w_tu([wsrc[j] for j in tch]))
    zk, zR = z(lk), z(R5)
    for nm_, v_ in (("lk", lk), ("R5", R5), ("zk", zk), ("zR", zR), ("l0", l0)):
        if not np.isfinite(v_).all():
            raise SystemExit(f"fold {i+1}: {nm_} khong huu han")
    e = {"fold": i + 1, "seed": seed, "nguon_chon": [src[j] + 1 for j in tch], "trongso_sha256": sha(ck)}
    e["ktn"] = float((ktn_dec == yt).mean()); e["ktn_numpy"] = hit(lk); e["ktnr"] = hit(zk + ALPHA * zR)
    e["ktnr_a025"] = hit(zk + 0.25 * zR); e["ktnr_a1"] = hit(zk + 1.0 * zR)
    rg = np.random.default_rng(seed + 992); S_ = len(src)
    e["ktnr_ngau5_lan"] = [hit(zk + ALPHA * z(Rof(w_tu([wsrc[j] for j in rg.choice(S_, K, replace=False)])))) for _ in range(R_DR)]
    e["ktnr_ngau5"] = float(np.mean(e["ktnr_ngau5_lan"]))
    e["ktnr_tatca"] = hit(zk + ALPHA * z(Rof(w_tu(wsrc))))
    if C == 9 or a.ds == "dong":                                         # Nakanishi: khong biet ten kenh nen bo bien the Oz
        oz = B.CH9.index("OZ") if C == 9 else DG.OZ; woz = np.zeros(C); woz[oz] = 1.0
        e["ktnr_oz"] = hit(zk + ALPHA * z(Rof(woz)))
    e["tinh"] = hit(l0); e["tinh_r"] = hit(z(l0) + ALPHA * zR)
    # D3 (them 2026-11-17, Dong 2023): KTN voi TB tham so 5 nguon ngau nhien, dung dinh nghia ngau5 cua b1moi_lai.py --d7
    # (rng(seed + 991), 20 lan, chon khong lap, mot me cho moi luot sau n0, argmax bang torch)
    rg7 = np.random.default_rng(seed + 991); lan7 = []
    with torch.no_grad():
        for _ in range(R_DR):
            js_ = rg7.choice(len(mods), K, replace=False)
            m7 = {k: torch.stack([mods[int(j)][k] for j in js_]).mean(0) for k in mc}
            lan7.append(float((L.sau(model, h1u[N0:], m7).argmax(1).cpu().numpy() == yt).mean()))
    e["ktn_ngau5_lan"] = lan7; e["ktn_ngau5"] = float(np.mean(lan7))
    e["ktn_ens5"] = float((lgs[top][:, N0:].mean(0).argmax(1).cpu().numpy() == yt).mean())   # = d7_ens5 cua b1moi_lai.py
    e["r_rieng"] = hit(R5)
    mk, mR = le(zk, yt), le(zR, yt)
    e["rho"] = float(np.corrcoef(mk, mR)[0, 1]) if mk.std() > 0 and mR.std() > 0 else None
    e["luot"] = dict(chi_so=list(range(N0 + 1, N0 + 1 + len(yt))), y=[int(v) for v in yt],       # diem z tung luot (chi so luot tu 1)
                     z_ktn=np.round(zk, 6).tolist(), z_R=np.round(zR, 6).tolist())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "eld", "nak", "dong"])
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--cho-phep-cpu", action="store_true")
    ap.add_argument("--thu", action="store_true", help="chay thu thong voi trong so 1 vong (bo kiem cau hinh huan luyen)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    ds = a.ds; DE.dang_ky(B, GD, ds); DN.dang_ky(B, GD, ds); DG.dang_ky(B, GD, ds); NT = GD.NT; nl = {**DE.SO_LUOT, **DN.SO_LUOT, **DG.SO_LUOT}[ds]; CK = {**DN.SO_KENH, **DG.SO_KENH}.get(ds, 9); c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))
    assert cfg["nch"] == L.NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg["subjects"] == 0 and cfg.get("ds", "benchmark") == ds
    if ds in ("nak", "dong") and not a.thu:                                           # mang nen dung cau hinh lo 2
        assert cfg["epochs"] == 1000 and cfg["lo"] == 100 and abs(cfg["dropout_cuoi"] - 0.95) < 1e-9 and cfg["seed"] == SEED and cfg.get("du_lieu_%s_sha256" % ds) == sha({"nak": DN, "dong": DG}[ds].__file__)
    D = {"X": [], "y": []}; freqs = None
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and X.shape[1] == CK and np.isfinite(X).all()                           # nhu b1moi_lai.py dong 217
        assert np.array_equal(np.bincount(y, minlength=NT), np.full(NT, nl // NT))
        freqs = np.asarray(fr, float) if freqs is None else freqs
        assert np.allclose(freqs, fr)
        Xp = GD.tien_xu_ly(X, ds, "causal", ns, fs); assert np.isfinite(Xp).all()
        D["X"].append(Xp); D["y"].append(y)
    assert len(freqs) == NT
    for i in range(a.folds or nsub):
        assert os.path.exists(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")), i + 1
    t0 = time.time(); per = []
    for i in range(a.folds or nsub):
        per.append(chay_fold(i, D, a, dev, freqs, fs, NT))
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        r = per[-1]
        print(f"fold {i+1}: ktn {r['ktn']:.3f} ktnr {r['ktnr']:.3f} | {time.time()-t0:.0f} s", flush=True)
    me = os.path.abspath(__file__)
    mac_nen = GD.dem_mac(None, CK, ns, nch=L.NCH)["tong"]
    js = dict(seed=SEED, ds=ds, T=a.T, n0=N0, k=K, alpha=ALPHA, H=H, r_dr=R_DR, nch=L.NCH, NT=NT, fs=fs, ns=ns, freqs=list(map(float, freqs)),
              trongso=a.trongso, trongso_script_sha256=cfg["script_sha256"], script_sha256=sha(me),
              ma_phu_sha256={os.path.basename(m_.__file__): sha(m_.__file__) for m_ in (B, GD, DE, L, DN, DG)},
              mac_du_toan=dict(mang_nen=mac_nen, nhanh_R=CK * ns + NT * 2 * H * ns + ns + 2 * H * NT, z_gop=6 * NT,
                               ghi_chu="du toan; HB3 dem tuong minh rieng"),
              tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)              # ghi nguyen tu
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
