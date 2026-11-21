#!/usr/bin/env python
"""B2, doi chung he so (bo sung 2026-11-20): cap nhat tung luot (EGM-R = E4R-b) so voi hai cach dat lai he so mot lan o luot 40.
  A1 (tinh theo do tin): lam_i ti le voi do tin TB tren 40 luot khoi dong cua 5 nguon da chon (g dung de chon nguon); g_min <= 0 thi dich de min = 0,01.
  A2 (khop lo): cung mat mat -do_tin(z(logit) + ALPHA z(R)) va buoc gradient mu cua E4R-b (b2e4r.phat_lai), chay vong qua 40 luot khoi dong
      (thu tu 0..39 lap lai) cho du T = n - N0 buoc, roi giu lam co dinh tu luot 41.
Cung nen UM-R, cung 5 nguon, cung w_R, cung luot cham 41..het. Dung lai ham cua b2e4r.py, khong sua tep do; phan dung mo-dun nguon chep
dung thu tu goi RNG cua b2e4r.chay_fold.
Cong hoi quy (moi fold, lech thi dung): nguon chon, quyet dinh KTNR va E4Rb tung luot trung JSON b2e4r (--json-b2e4r); UM-R tinh lai bang
gop(lam deu) trung quyet dinh KTNR o moi luot co bien > 1e-6.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2e4r as E               # noqa: E402
G, L, B, GD, DL = E.G, E.L, E.B, E.GD, E.DL
SEED, N0, NCH, K5, ALPHA = E.SEED, E.N0, E.NCH, E.K5, E.ALPHA
sha = E.sha


def quyet_dinh(model, h1, DW, DB, lam, zR, dev):
    lv = torch.tensor(lam, device=dev, dtype=torch.float32)
    with torch.no_grad():
        lg = L.sau(model, h1, G.gop(DW, DB, lv)).double().cpu().numpy()
    return (E.z(lg) + ALPHA * zR).argmax(1), lg


def khop_lo(model, h1u, DW, DB, zR_all, T, dev):
    """Gradient mu nhu b2e4r.phat_lai (day_gop = True) nhung luot thu k lay tu 40 luot khoi dong (k mod N0)."""
    K = DW.shape[0]; theta = torch.zeros(K, device=dev); Gm = 0.0; eta = 0.0; bo = 0
    zRt = torch.tensor(zR_all, device=dev, dtype=torch.float32)
    for k in range(T):
        t = k % N0
        lv = torch.softmax(theta, 0).requires_grad_(True)
        lg = L.sau(model, h1u[t:t + 1], G.gop(DW, DB, lv))
        dt = L.do_tin(E.z_t(lg) + ALPHA * zRt[t:t + 1])
        (g,) = torch.autograd.grad(-dt.sum(), lv)
        if not torch.isfinite(g).all():
            bo += 1; continue
        Gm = max(Gm, float(g.abs().max()))
        if Gm > 0:
            eta = math.sqrt(2.0 * math.log(K) / (Gm * Gm * T))
            theta = theta - eta * g
    return torch.softmax(theta, 0).cpu().numpy(), dict(G=Gm, eta_cuoi=eta, bo_khong_huu_han=bo)


def chay_fold(i, D, a, dev, freqs, fs, NT, gb):
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))                  # nhu b2e4r.chay_fold
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    assert gb["seed"] == seed and gb["fold"] == i + 1
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95).to(dev)
    ck = os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")
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
    with torch.no_grad():                                                # logit KTN nhu b2e4r (me LO tu N0)
        for a_ in range(N0, h1u.shape[0], L.LO):
            lk_t.append(L.sau(model, h1u[a_:a_ + L.LO], {k: v.clone() for k, v in ms.items()}))
    lk = torch.cat(lk_t).double().cpu().numpy()
    Ua = h1u[:, 0].double().cpu().numpy(); Ua = Ua - Ua.mean(-1, keepdims=True)
    Ra = E.diem_R(np.einsum("c,ict->it", E.w_tu([wsrc[int(j)] for j in top]), Ua), E.co_so(freqs, fs, ns))
    zR_all = E.z(Ra); zR = zR_all[N0:]
    d_ktnr = (E.z(lk) + ALPHA * zR).argmax(1)
    assert d_ktnr.tolist() == gb["dec"]["KTNR"], f"hoi quy KTNR lech o fold {i+1}"
    eb, lgb, thb = E.phat_lai(model, h1u, lg_c, DW5, DB5, zR, ALPHA, True)
    d_b = (E.z(lgb) + ALPHA * zR).argmax(1)
    assert d_b.tolist() == gb["dec"]["E4Rb"], f"hoi quy E4Rb lech o fold {i+1}"
    lam_u = np.full(K5, 1.0 / K5)
    d_u, lg_u = quyet_dinh(model, h1u[N0:], DW5, DB5, lam_u, zR, dev)
    s_u = E.z(lg_u) + ALPHA * zR; ss = np.sort(s_u, 1); bien = ss[:, -1] - ss[:, -2]
    lech_u = int(((d_u != d_ktnr) & (bien > 1e-6)).sum())
    assert lech_u == 0, f"gop(lam deu) khac KTNR o {lech_u} luot, fold {i+1}"
    gt = g[top].double().cpu().numpy()
    gt1 = gt - gt.min() + 0.01 if gt.min() <= 0 else gt
    lam1 = gt1 / gt1.sum()
    d_1, _ = quyet_dinh(model, h1u[N0:], DW5, DB5, lam1, zR, dev)
    T = h1u.shape[0] - N0
    lam2, th2 = khop_lo(model, h1u, DW5, DB5, zR_all, T, dev)
    d_2, _ = quyet_dinh(model, h1u[N0:], DW5, DB5, lam2, zR, dev)
    acc = lambda d: float((d == yt).mean())
    return dict(fold=i + 1, seed=seed, nguon_chon5=chon, n_cham=int(len(yt)), g_top=gt.tolist(), dich_g=bool(gt.min() <= 0),
                KTNR=acc(d_ktnr), E4Rb=acc(d_b), A1=acc(d_1), A2=acc(d_2),
                lam_b=thb["lam"], tv_b=thb["tv"], lam_A1=lam1.tolist(), tv_A1=G.tv(lam1), lam_A2=lam2.tolist(), tv_A2=G.tv(lam2),
                G_A2=th2["G"], eta_cuoi_A2=th2["eta_cuoi"], bo_khong_huu_han_A2=th2["bo_khong_huu_han"],
                P4_A1=float((d_1 != d_ktnr).mean()), P4_A2=float((d_2 != d_ktnr).mean()), P4_b=float((d_b != d_ktnr).mean()),
                dec=dict(A1=[int(v) for v in d_1], A2=[int(v) for v in d_2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "dong"])
    ap.add_argument("--T", type=float, required=True, choices=[0.6, 1.0])
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--json-b2e4r", required=True, help="JSON ket qua b2e4r.py cung bo, cung T (cong hoi quy)")
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
    assert os.path.normpath(gb["trongso"]) == os.path.normpath(a.trongso) or os.path.basename(gb["trongso"]) == os.path.basename(a.trongso)
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
        print(f"fold {i+1} chay thong {time.time()-t0:.0f} s", flush=True)      # khong in do chinh xac trong luc chay
    js = dict(seed=SEED, ds=ds, T=a.T, n0=N0, alpha=ALPHA, k=K5, thiet_bi=str(dev), numpy=np.__version__, torch=torch.__version__,
              script_sha256=sha(os.path.abspath(__file__)), b2e4r_sha256=sha(E.__file__),
              b1gop_sha256=sha(G.__file__), b1moi_lai_sha256=sha(L.__file__), global_dnn_sha256=sha(GD.__file__),
              json_b2e4r=a.json_b2e4r, json_b2e4r_sha256=sha(a.json_b2e4r), json_ktnr=a.json_ktnr, json_ktnr_sha256=sha(a.json_ktnr),
              trongso=a.trongso, du_fold=(len(folds) == nsub), tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"))
    os.link(a.out + ".tmp", a.out); os.remove(a.out + ".tmp")          # os.link that bai neu tep ket qua da co
    if os.path.exists(a.out + ".dang-chay.tmp"):
        os.remove(a.out + ".dang-chay.tmp")
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
