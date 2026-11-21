#!/usr/bin/env python
"""Phat lai DH-TT: nhanh A0, A (alpha 1e-5), A* (alpha 1e-4, 1e-3), C.
Cham luot 41..het (chi so 40..). Khong sua b2e4r.py; cac buoc dung UM-R (KTN-R) chep tu b2e4r.chay_fold
(dong 123-168) va kiem bang cung cong hoi quy (nguon chon, z_ktn, z_R, do chinh xac KTN-R so voi JSON KTN-R).
  A0, A, A*: W0^EA (dh_tt_huanluyen.py, --ea co --tau 2) + EA truc tuyen tu luot 1, cap nhat moi tham so, alpha in {0, 1e-5, 1e-4, 1e-3}.
  C        : mang UM-R (W0 voi l2 + TB so gia 5 nguon) KHONG EA, tu luot 41 voi F = 0, cap nhat moi tham so; nhanh R tu l1 DONG BANG
             cua UM-R; quyet dinh = argmax[z(logit) + 0,5 z(R)] (numpy float64 nhu b2e4r); alpha in {0, 1e-5, 1e-4, 1e-3}.
Cong (dung neu lech): C alpha 0 trung KTN-R o moi luot chung nhan (bien > tol) va thuoc nhom sat dau o luot con lai.
Chi so co che: P4 (A_alpha so A0, luot 41..), chuan gradient TB, do dich logit TB |z - z_A0|, ti le q_max > gamma, tong F cuoi, so bo.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1gop as G            # noqa: E402
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402
import b1moi_lai as L        # noqa: E402
import du_lieu_lee as DL     # noqa: E402
import b2e4r as E            # noqa: E402  (chi dung ham thuan trca, w_tu, co_so, diem_R, z, hang so)
import dh_tt_loi as DH       # noqa: E402

SEED, N0, NCH, K5, ALPHA_R = G.SEED, G.N0, G.NCH, G.K_CHINH, E.ALPHA
ALPHAS = (0.0,) + DH.ALPHAS
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def tom(r, ref_lg, yt, tu=N0, dec=None):
    """Tom tat mot lan phat lai: do chinh xac luot tu.. (dec mac dinh r['dec']), chi so co che so voi logit tham chieu."""
    d = r["dec"][tu:] if dec is None else dec
    assert np.isfinite(r["logit"][tu:]).all() and (d >= 0).all()
    out = dict(acc=float((d == yt).mean()), vuot_gamma=int(r["vuot_gamma"]), so_luot_xu_ly=int(r["so_luot"]),
               ti_le_vuot_gamma=float(r["vuot_gamma"] / r["so_luot"]), F=[float(v) for v in r["F"]], F_max=float(r["F"].max()),
               chuan_tb=r["chuan_tb"], bo_khong_huu_han=int(r["bo_khong_huu_han"]))
    if ref_lg is not None:
        out["dich_logit_tb"] = float(np.abs(r["logit"][tu:] - ref_lg[tu:]).mean())
    return out


def chay_fold(i, D, a, dev, freqs, fs, NT, gk):
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    assert gk["seed"] == seed
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    y = D["y"][i]; yt = y[N0:]
    res = dict(fold=i + 1, seed=seed, n_cham=int(len(yt)))
    # ---------- A0, A, A* ----------
    mea = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95)
    mea.load_state_dict(torch.load(os.path.join(a.trongso_ea, f"fold{i+1:03d}_trongso.pt"), map_location="cpu")); mea.eval()
    Rt = np.load(os.path.join(a.trongso_ea, f"fold{i+1:03d}_Rtrain.npy"))
    Rt2 = DH.r_nguon(np.concatenate([D["X"][s] for s in src]))               # chong ro ri: R_train tinh lai chi tu nguoi nguon phai trung tep
    assert np.allclose(Rt, Rt2, rtol=1e-10, atol=1e-12), f"R_train lech o fold {i+1}"
    A = {}
    for al in ALPHAS:
        torch.manual_seed(seed); np.random.seed(seed)                    # dat lai RNG cho moi nhanh, moi alpha
        A[al] = DH.phat_lai_dh(mea, D["X"][i], al, dev, R_train=Rt)
    lg0 = A[0.0]["logit"]
    res["A"] = {}
    for al in ALPHAS:
        t_ = tom(A[al], lg0 if al > 0 else None, yt)
        t_["acc_ca_phien"] = float((A[al]["dec"] == y).mean())
        t_["P4_so_A0"] = float((A[al]["dec"][N0:] != A[0.0]["dec"][N0:]).mean())
        t_["dung_theo_luot"] = (A[al]["dec"] == y).astype(int).tolist()
        res["A"][repr(al)] = t_
    del A
    # ---------- dung UM-R (chep b2e4r.chay_fold) ----------
    model = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95).to(dev)
    ck = os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")
    assert sha(ck) == gk["trongso_sha256"], f"trong so fold {i+1} khac JSON KTN-R"
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev)
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():
        h1u = L.truoc(model, Xu); lg_c = L.sau(model, h1u, mc)
        assert torch.allclose(model(Xu), lg_c, atol=1e-4)
    mods, wsrc = [], []
    for s in src:
        Xs = torch.tensor(D["X"][s], device=dev); ys = torch.tensor(D["y"][s], device=dev, dtype=torch.long)
        with torch.no_grad():
            h1s = L.truoc(model, Xs)
        mods.append(L.hoc(model, h1s, ys, mc, L.BUOC_NGUON, L.LR_MOD))
        Us = h1s[:, 0].double().cpu().numpy(); Us = Us - Us.mean(-1, keepdims=True)
        wsrc.append(E.trca(Us, D["y"][s], NT))
    with torch.no_grad():
        lgs = torch.stack([L.sau(model, h1u, m) for m in mods])
    g = L.do_tin(lgs[:, :N0]).mean(1)
    top = torch.argsort(-g, stable=True)[:K5]
    chon = [src[int(j)] + 1 for j in top]
    assert chon == gk["nguon_chon"], f"nguon chon lech o fold {i+1}"
    ms = {k: torch.stack([mods[int(j)][k] for j in top]).mean(0) for k in mc}
    lk_t = []
    with torch.no_grad():
        for a_ in range(N0, h1u.shape[0], L.LO):
            lk_t.append(L.sau(model, h1u[a_:a_ + L.LO], {k: v.clone() for k, v in ms.items()}))
    lk = torch.cat(lk_t).double().cpu().numpy()
    U = h1u[N0:, 0].double().cpu().numpy(); U = U - U.mean(-1, keepdims=True)
    R5 = E.diem_R(np.einsum("c,ict->it", E.w_tu([wsrc[int(j)] for j in top]), U), E.co_so(freqs, fs, ns))
    zk, zR = E.z(lk), E.z(R5)
    lch_k = float(np.abs(zk - np.array(gk["luot"]["z_ktn"])).max()); lch_R = float(np.abs(zR - np.array(gk["luot"]["z_R"])).max())
    assert lch_k < 1e-5 and lch_R < 1e-5, f"hoi quy z lech o fold {i+1}"
    d_ktnr = (zk + ALPHA_R * zR).argmax(1)
    assert float((d_ktnr == yt).mean()) == gk["ktnr"], f"hoi quy KTNR lech o fold {i+1}"
    assert gk["luot"]["y"] == [int(v) for v in yt] and gk["luot"]["chi_so"] == list(range(N0 + 1, N0 + 1 + len(yt))), f"anh xa luot lech o fold {i+1}"
    res["W0"] = float((lg_c[N0:].argmax(1).cpu().numpy() == yt).mean())        # do chinh xac mang tinh W0, luot 41..
    # ---------- C: mang UM-R gop vao l2, khong EA, tu luot 41 ----------
    mum = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95)
    mum.load_state_dict(torch.load(ck, map_location="cpu"))
    with torch.no_grad():
        mum.l2.weight += ms["dW"].cpu(); mum.l2.bias += ms["db"].cpu()
    mum.eval()
    res["C"] = {}; dC0 = None; lgC0 = None
    for al in ALPHAS:
        torch.manual_seed(seed); np.random.seed(seed)
        r = DH.phat_lai_dh(mum, D["X"][i], al, dev, R_train=None, t0=N0)
        lgr = r["logit"][N0:]
        dC = (E.z(lgr) + ALPHA_R * zR).argmax(1)
        if al == 0.0:                                                        # cong: C alpha 0 = UM-R
            ref = zk + ALPHA_R * zR; cand = E.z(lgr) + ALPHA_R * zR
            lch = float(np.abs(cand - ref).max()); tol = 2 * lch + 1e-12
            srt = np.sort(ref, 1); cn = (srt[:, -1] - srt[:, -2]) > tol
            # nguong diem 1e-3 (sua 2026-11-19 sau job 18873/18874/18876): UM-R tinh logit theo lo 20 luot, C tung luot; cuDNN khac thuat toan
            # theo co lo cho sai so float32 ~1e-6 o logit, phep z khuech dai len ~1e-4 (94 fold qua nguong cu: lech toi da 9,6e-5, 0 quyet dinh khac).
            assert lch < 1e-3 and np.array_equal(dC[cn], d_ktnr[cn]), f"cong C alpha 0 = UM-R lech o fold {i+1} (lech {lch:.2e})"
            gan = ref >= (srt[:, -1:] - tol)
            assert all(gan[k_, dC[k_]] for k_ in np.where(~cn)[0])
            res["cong_C0"] = dict(lech_diem_toi_da=lch, so_luot_khong_chung_nhan=int((~cn).sum()), so_quyet_dinh_khac_UMR=int((dC != d_ktnr).sum()))
            dC0, lgC0 = dC, r["logit"]
        t_ = tom(r, lgC0 if al > 0 else None, yt, dec=dC)
        t_["P4_so_C0"] = float((dC != dC0).mean())
        res["C"][repr(al)] = t_
    res["UMR"] = float((d_ktnr == yt).mean()); res["nguon_chon"] = chon
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "dong"])
    ap.add_argument("--T", type=float, required=True, choices=[0.6, 1.0])
    ap.add_argument("--trongso", required=True, help="W0 (cung thu muc b2e4r da dung)")
    ap.add_argument("--trongso-ea", required=True, help="W0^EA (dh_tt_huanluyen.py --ea co --tau 2)")
    ap.add_argument("--json-ktnr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all", help="vi du 1-2")
    ap.add_argument("--thiet-bi", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    if a.thiet_bi == "cuda" and not torch.cuda.is_available():
        sys.exit("khong co CUDA")
    dev = torch.device(a.thiet_bi)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    ds = a.ds
    if ds == "dong":
        import du_lieu_dong as DG
        DG.dang_ky(B, GD, ds); nl = DG.SO_LUOT[ds]; CK = DG.SO_KENH[ds]
    else:
        DL.dang_ky(B, GD, ds); nl = DL.SO_LUOT[ds]; CK = 9
    c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    gk = json.load(open(a.json_ktnr))
    assert gk["ds"] == ds and abs(gk["T"] - a.T) < 1e-9 and gk["alpha"] == ALPHA_R and gk["k"] == K5 and gk["n0"] == N0
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))
    assert cfg["nch"] == NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg.get("ds", "benchmark") == ds
    assert cfg["script_sha256"] == gk["trongso_script_sha256"]
    ce = json.load(open(os.path.join(a.trongso_ea, "config.json")))
    assert ce["ea"] == "co" and ce["tau"] == DH.TAU and ce["ds"] == ds and abs(ce["T"] - a.T) < 1e-9
    assert ce["dh_tt_loi_sha256"] == sha(DH.__file__), "dh_tt_loi.py khac ban da dung khi huan luyen W0^EA"
    assert (ce["epochs"], ce["lo"], ce["nch"], ce["seed"], ce["loc"], ce["subjects"]) == (1000, 100, NCH, SEED, "causal", 0) and abs(ce["dropout_cuoi"] - 0.95) < 1e-9, "W0^EA khac cong thuc W0"
    assert ce["script_sha256"] == sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dh_tt_huanluyen.py")), "W0^EA khong do dh_tt_huanluyen.py hien tai huan luyen"
    folds = list(range(nsub)) if a.folds == "all" else list(range(int(a.folds.split("-")[0]) - 1, int(a.folds.split("-")[1])))
    assert folds and all(0 <= f_ < nsub for f_ in folds), "--folds ngoai mien"
    for f_ in folds:
        assert os.path.exists(os.path.join(a.trongso_ea, f"fold{f_ + 1:03d}.json")), f"W0^EA thieu fold {f_ + 1}"
    D = {"X": [], "y": []}; freqs = None
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and X.shape[1] == CK and np.isfinite(X).all() and np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, nl // GD.NT))
        freqs = np.asarray(fr, float) if freqs is None else freqs
        assert np.allclose(freqs, fr)
        D["X"].append(GD.tien_xu_ly(X, ds, "causal", ns, fs)); D["y"].append(y)
    assert np.allclose(freqs, gk["freqs"]) and ns == gk["ns"] and fs == gk["fs"]
    t0 = time.time(); per = []
    for i in folds:
        per.append(chay_fold(i, D, a, dev, freqs, fs, GD.NT, gk["tung_nguoi"][i]))
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"fold {i+1} chay thong {time.time()-t0:.0f} s", flush=True)      # khong in do chinh xac
    js = dict(seed=SEED, ds=ds, T=a.T, n0=N0, alphas=list(ALPHAS), hang_so=dict(tau=DH.TAU, n0_ea=DH.N0_EA, omega=DH.OMEGA, gamma=DH.GAMMA,
              beta=DH.BETA, lam=DH.LAMBDA, kep=DH.KEP), thiet_bi=str(dev), torch=torch.__version__, numpy=np.__version__,
              script_sha256=sha(os.path.abspath(__file__)), dh_tt_loi_sha256=sha(DH.__file__),
              b2e4r_sha256=sha(E.__file__), b1moi_lai_sha256=sha(L.__file__), global_dnn_sha256=sha(GD.__file__),
              root_du_lieu=B.DS[ds]["root"], trongso=a.trongso, trongso_ea=a.trongso_ea, json_ktnr=a.json_ktnr, json_ktnr_sha256=sha(a.json_ktnr),
              trongso_ea_fold_sha256=[sha(os.path.join(a.trongso_ea, f"fold{i+1:03d}_trongso.pt")) for i in folds],
              du_fold=(len(folds) == nsub), tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"))
    os.link(a.out + ".tmp", a.out); os.remove(a.out + ".tmp")
    if os.path.exists(a.out + ".dang-chay.tmp"):
        os.remove(a.out + ".dang-chay.tmp")
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
