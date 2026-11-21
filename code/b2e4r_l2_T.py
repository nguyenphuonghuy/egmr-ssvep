#!/usr/bin/env python
"""B2, phan tich bo sung 2026-11-20 (F, G, H), cung nen UM-R, cung 5 nguon, cung w_R, cung luot cham 41..het.
  F: cap nhat tung luot 220 tham so l2 (dW, db) voi cung mat mat -do_tin(z(logit) + ALPHA z(R)), bat dau tu TB 5 so gia, buoc co dinh eta.
  G: EGM-R (b2e4r.phat_lai) voi T trong cong thuc buoc thay bang 0,5 T, 2 T, 1000.
  H: do chinh xac cua mang chung tren 40 luot dau.
Phan dung mo-dun nguon chep tu b2e4r_dclam.py (cung thu tu goi RNG). Cong hoi quy nhu b2e4r_dclam.py: nguon chon, quyet dinh KTNR va E4Rb
tung luot trung JSON b2e4r; G voi he so T = 1 tai lap dung E4Rb.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2e4r as E               # noqa: E402
G, L, B, GD, DL = E.G, E.L, E.B, E.GD, E.DL
SEED, N0, NCH, K5, ALPHA = E.SEED, E.N0, E.NCH, E.K5, E.ALPHA
sha = E.sha
ETA_L2 = (1e-4, 1e-3, 1e-2)
T_THAY = (("0.5T", 0.5, None), ("2T", 2.0, None), ("T1000", None, 1000))


def eg_T(model, h1, DW, DB, zR, Tdung):
    """Nhu b2e4r.phat_lai(day_gop = True) nhung T trong cong thuc buoc la Tdung. Tra quyet dinh tung luot N0.. va lam cuoi."""
    K = DW.shape[0]; n = h1.shape[0]; dev = h1.device
    theta = torch.zeros(K, device=dev); Gm = 0.0; lgs = []
    zRt = torch.tensor(zR, device=dev, dtype=torch.float32)
    for t in range(N0, n):
        lv = torch.softmax(theta, 0).requires_grad_(True)
        lg = L.sau(model, h1[t:t + 1], G.gop(DW, DB, lv))
        lgs.append(lg.detach().double().cpu().numpy()[0])
        dt = L.do_tin(E.z_t(lg) + ALPHA * zRt[t - N0:t - N0 + 1])
        (g,) = torch.autograd.grad(-dt.sum(), lv)
        if not torch.isfinite(g).all():
            continue
        Gm = max(Gm, float(g.abs().max()))
        if Gm > 0:
            theta = theta - math.sqrt(2.0 * math.log(K) / (Gm * Gm * Tdung)) * g
    return (E.z(np.array(lgs)) + ALPHA * zR).argmax(1), torch.softmax(theta, 0).cpu().numpy()


def l2_tung_luot(model, h1, ms, zR, eta):
    """Cap nhat truc tiep dW, db (220 + 22 tham so) sau moi quyet dinh, bat dau tu TB 5 so gia (UM-R)."""
    dW = ms["dW"].clone(); db = ms["db"].clone(); n = h1.shape[0]; dev = h1.device; lgs = []; bo = 0
    zRt = torch.tensor(zR, device=dev, dtype=torch.float32)
    for t in range(N0, n):
        dW.requires_grad_(True); db.requires_grad_(True)
        lg = L.sau(model, h1[t:t + 1], {"dW": dW, "db": db})
        lgs.append(lg.detach().double().cpu().numpy()[0])
        dt = L.do_tin(E.z_t(lg) + ALPHA * zRt[t - N0:t - N0 + 1])
        gW, gb = torch.autograd.grad(-dt.sum(), [dW, db])
        dW = dW.detach(); db = db.detach()
        if not (torch.isfinite(gW).all() and torch.isfinite(gb).all()):
            bo += 1; continue
        dW = dW - eta * gW; db = db - eta * gb
    d = float(torch.sqrt(((dW - ms["dW"]) ** 2).sum() + ((db - ms["db"]) ** 2).sum()))
    return (E.z(np.array(lgs)) + ALPHA * zR).argmax(1), d, bo


def chay_fold(i, D, a, dev, freqs, fs, NT, gb):
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))                  # nhu b2e4r.chay_fold
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    assert gb["seed"] == seed and gb["fold"] == i + 1
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt"), map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev); y = D["y"][i]; yt = y[N0:]
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():
        h1u = L.truoc(model, Xu); lg0 = model(Xu)
        lg_c = L.sau(model, h1u, mc)
        assert torch.allclose(lg0, lg_c, atol=1e-4)
    mods, wsrc = [], []
    for s in src:                                                        # cung thu tu goi nhu b2e4r.chay_fold
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
    assert chon == gb["nguon_chon5"], f"nguon chon lech o fold {i+1}"
    DW5, DB5 = G.xep(mods, top)
    ms = {k: torch.stack([mods[int(j)][k] for j in top]).mean(0) for k in mc}
    lk_t = []
    with torch.no_grad():
        for a_ in range(N0, h1u.shape[0], L.LO):
            lk_t.append(L.sau(model, h1u[a_:a_ + L.LO], {k: v.clone() for k, v in ms.items()}))
    lk = torch.cat(lk_t).double().cpu().numpy()
    U = h1u[N0:, 0].double().cpu().numpy(); U = U - U.mean(-1, keepdims=True)
    zR = E.z(E.diem_R(np.einsum("c,ict->it", E.w_tu([wsrc[int(j)] for j in top]), U), E.co_so(freqs, fs, ns)))
    d_ktnr = (E.z(lk) + ALPHA * zR).argmax(1)
    assert d_ktnr.tolist() == gb["dec"]["KTNR"], f"hoi quy KTNR lech o fold {i+1}"
    T = h1u.shape[0] - N0
    d_b, lam_b = eg_T(model, h1u, DW5, DB5, zR, T)
    assert d_b.tolist() == gb["dec"]["E4Rb"], f"hoi quy E4Rb (eg_T voi T goc) lech o fold {i+1}"
    acc = lambda d: float((d == yt).mean())
    res = dict(fold=i + 1, seed=seed, nguon_chon5=chon, n_cham=int(len(yt)), T=T, KTNR=acc(d_ktnr), E4Rb=acc(d_b),
               tinh_40_dau=float((lg_c[:N0].argmax(1).cpu().numpy() == y[:N0]).mean()), l2={}, T_thay={})
    for eta in ETA_L2:
        d, dd, bo = l2_tung_luot(model, h1u, ms, zR, eta)
        res["l2"]["%g" % eta] = dict(acc=acc(d), P4=float((d != d_ktnr).mean()), dich_l2=dd, bo_khong_huu_han=bo)
    for ten, he_so, co_dinh in T_THAY:
        Td = co_dinh if co_dinh else he_so * T
        d, lam = eg_T(model, h1u, DW5, DB5, zR, Td)
        res["T_thay"][ten] = dict(acc=acc(d), T_dung=Td, tv=G.tv(lam), P4=float((d != d_ktnr).mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "dong"])
    ap.add_argument("--T", type=float, required=True, choices=[0.6, 1.0])
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--json-b2e4r", required=True)
    ap.add_argument("--json-ktnr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all")
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
        DG.dang_ky(B, GD, ds); nl = DG.SO_LUOT[ds]
    else:
        DL.dang_ky(B, GD, ds); nl = DL.SO_LUOT[ds]
    c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    gb = json.load(open(a.json_b2e4r)); gk = json.load(open(a.json_ktnr))
    assert gb["ds"] == ds and abs(gb["T"] - a.T) < 1e-9 and gb["du_fold"] and gb["alpha"] == ALPHA and gb["k"] == K5 and gb["n0"] == N0
    assert [p["fold"] for p in gb["tung_nguoi"]] == list(range(1, nsub + 1))
    folds = list(range(nsub)) if a.folds == "all" else list(range(int(a.folds.split("-")[0]) - 1, int(a.folds.split("-")[1])))
    for i in folds:
        assert sha(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")) == gb["trongso_fold_sha256"][i], f"trong so fold {i+1} khac JSON b2e4r"
    D = {"X": [], "y": []}; freqs = None
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and np.isfinite(X).all()
        freqs = np.asarray(fr, float) if freqs is None else freqs
        D["X"].append(GD.tien_xu_ly(X, ds, "causal", ns, fs)); D["y"].append(y)
    assert np.allclose(freqs, gk["freqs"]) and ns == gk["ns"] and fs == gk["fs"]
    print("thiet bi:", dev, "| numpy", np.__version__, "| torch", torch.__version__, flush=True)
    t0 = time.time(); per = []
    for i in folds:
        per.append(chay_fold(i, D, a, dev, freqs, fs, GD.NT, gb["tung_nguoi"][i]))
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"fold {i+1} chay thong {time.time()-t0:.0f} s", flush=True)
    js = dict(seed=SEED, ds=ds, T=a.T, n0=N0, alpha=ALPHA, k=K5, thiet_bi=str(dev), numpy=np.__version__, torch=torch.__version__,
              eta_l2=list(ETA_L2), T_thay=[t[0] for t in T_THAY],
              script_sha256=sha(os.path.abspath(__file__)), b2e4r_sha256=sha(E.__file__), b1gop_sha256=sha(G.__file__),
              b1moi_lai_sha256=sha(L.__file__), global_dnn_sha256=sha(GD.__file__), json_b2e4r=a.json_b2e4r, json_b2e4r_sha256=sha(a.json_b2e4r),
              json_ktnr_sha256=sha(a.json_ktnr), trongso=a.trongso, du_fold=(len(folds) == nsub), tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"))
    os.link(a.out + ".tmp", a.out); os.remove(a.out + ".tmp")
    if os.path.exists(a.out + ".dang-chay.tmp"):
        os.remove(a.out + ".dang-chay.tmp")
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
