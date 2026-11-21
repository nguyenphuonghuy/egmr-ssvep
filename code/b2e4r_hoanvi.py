#!/usr/bin/env python
"""B2, phu luc 1: do ben cua E4R-b (so voi KTN-R) theo thu tu phat lai luot.
Ghep b1gop_hoanvi.py (hoan vi, cong bat bien) voi b2e4r.py (nhanh R, E4R-b); dung lai ham cua b2e4r.py, khong sua tep do.
Ho P1 (chinh): hoan vi trong tung khoi; ho P2 (bao kem): hoan vi toan phien. Seed hoan vi np.random.default_rng([20261107, i, r, P]) (nhu thi nghiem A).
r = 0 la dong nhat: CONG HOI QUY tai lap tung quyet dinh KTNR, E4Rb cua JSON G0/G1 (b2e4r.py, --json-b2e4r); lech thi dung.
P1: 5 nguon, w_R, z(R) tung luot va quyet dinh KTN-R tung luot bat bien; cong bat bien moi hoan vi: top-5 trung, KTNR_perm[j] == KTNR_goc[pi[j]].
P2: chon lai 5 nguon theo pi[:40] (sap xep tang), dung lai w_R tu 5 nguon moi, tinh lai logit KTN (me LO tu N0 nhu ktnr_buoc1) va R cho cac luot pi[40:].
E4R-b: t, T, G_t, eta_t chay theo vi tri phat lai; luot thu t lay tu chi so goc pi[t]; z(R) cua luot do di kem.
Them 2026-11-20: --ds dong, nap bang du_lieu_dong.py; Benchmark, BETA giu nguyen.
Dung: python b2e4r_hoanvi.py --ds benchmark --T 1.0 --trongso ... --json-b2e4r ... --json-ktnr ... --out ... [--r-dau 0 --r-cuoi 20 --folds 1-2]
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2e4r as E               # noqa: E402  (nhanh R, phat_lai, z; nap b1gop, b1moi_lai)
G, L, B, GD, DL = E.G, E.L, E.B, E.GD, E.DL
SEED, N0, NCH, K5, ALPHA = E.SEED, E.N0, E.NCH, E.K5, E.ALPHA
sha = E.sha


def hoan_vi(i, r, P, nl, nb):                                            # ban sao nguyen van b1gop_hoanvi.hoan_vi
    if r == 0:
        return np.arange(nl)
    rng = np.random.default_rng([20261107, i, r, P])
    if P == 1:
        k = nl // nb
        return np.concatenate([b * k + rng.permutation(k) for b in range(nb)])
    return rng.permutation(nl)


def chay_fold(i, D, a, dev, freqs, fs, NT, gb, gk, ho, rs, nb):
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    assert gb["seed"] == seed and gk["seed"] == seed
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95).to(dev)
    ck = os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")
    assert sha(ck) == gk["trongso_sha256"], f"trong so fold {i+1} khac JSON KTN-R"
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev); y = D["y"][i]; nl = len(y)
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():
        h1u = L.truoc(model, Xu); lg_c = L.sau(model, h1u, mc)
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
    Qb = E.co_so(freqs, fs, ns)

    def chon(pi):
        idx0 = torch.tensor(np.sort(pi[:N0]), device=dev)
        return torch.argsort(-L.do_tin(lgs[:, idx0]).mean(1), stable=True)[:K5]

    def bo(top, idx_sau):
        """5 nguon -> DW, DB, zk, zR cho cac luot idx_sau (chi so goc, theo thu tu phat lai)."""
        DW, DB = G.xep(mods, top)
        ms = {k: torch.stack([mods[int(j)][k] for j in top]).mean(0) for k in mc}
        it = torch.tensor(idx_sau, device=dev); lk = []
        with torch.no_grad():
            for a_ in range(0, len(idx_sau), L.LO):
                lk.append(L.sau(model, h1u[it[a_:a_ + L.LO]], {k: v.clone() for k, v in ms.items()}))
        lk = torch.cat(lk).double().cpu().numpy()
        U = h1u[it, 0].double().cpu().numpy(); U = U - U.mean(-1, keepdims=True)
        R5 = E.diem_R(np.einsum("c,ict->it", E.w_tu([wsrc[int(j)] for j in top]), U), Qb)
        return DW, DB, E.z(lk), E.z(R5)

    top_goc = chon(np.arange(nl))
    chon_goc = [src[int(j)] + 1 for j in top_goc]
    assert chon_goc == gb["nguon_chon5"] == gk["nguon_chon"], f"nguon chon lech o fold {i+1}"
    DW0, DB0, zk0, zR0 = bo(top_goc, np.arange(N0, nl))
    lch = max(float(np.abs(zk0 - np.array(gk["luot"]["z_ktn"])).max()), float(np.abs(zR0 - np.array(gk["luot"]["z_R"])).max()))
    assert lch < 1e-5, f"hoi quy z lech o fold {i+1}: {lch:.2e}"
    ktnr0 = (zk0 + ALPHA * zR0).argmax(1)                               # theo chi so goc N0..nl-1
    assert [int(v) for v in ktnr0] == gb["dec"]["KTNR"], f"hoi quy KTNR lech o fold {i+1}"
    out = dict(fold=i + 1, seed=seed, nguon_chon5_goc=chon_goc, hoan_vi=[])
    for P in ho:
        for r in rs:
            if r == 0 and P != ho[0]:
                continue
            pi = hoan_vi(i + 1, r, P, nl, nb)
            pit = torch.tensor(pi, device=dev)
            yp = y[pi]
            top = chon(pi)
            if P == 1 or r == 0:
                assert torch.equal(top, top_goc), f"cong bat bien: top-5 doi o fold {i+1} P{P} r{r}"
                assert (pi[N0:] >= N0).all()
                DW, DB = DW0, DB0; zRp = zR0[pi[N0:] - N0]
                ktnr = ktnr0[pi[N0:] - N0]                               # cong bat bien: quyet dinh KTN-R tung luot khong doi
            else:
                DW, DB, zkp, zRp = bo(top, pi[N0:])
                ktnr = (zkp + ALPHA * zRp).argmax(1)
            dec_lg, lgb, th = E.phat_lai(model, h1u[pit], lg_c[pit], DW, DB, zRp, ALPHA, True)
            assert np.isfinite(lgb).all() and np.isfinite(th["lam"]).all(), f"khong huu han o fold {i+1} P{P} r{r}"
            eb = (E.z(lgb) + ALPHA * zRp).argmax(1)
            if r == 0:
                assert [int(v) for v in eb] == gb["dec"]["E4Rb"], f"hoi quy E4Rb lech o fold {i+1}"
            ys_ = yp[N0:]
            ban = dict(P=P, r=r, pi_sha256=hashlib.sha256(pi.astype(np.int64).tobytes()).hexdigest(), pi=[int(v) for v in pi],
                       E4Rb=float((eb == ys_).mean()), KTNR=float((ktnr == ys_).mean()), P4=float((eb != ktnr).mean()),
                       lam=th["lam"], tv=th["tv"], G=th["G"], eta_cuoi=th["eta_cuoi"], bo_khong_huu_han=th["bo_khong_huu_han"], bo_G0=th["bo_G0"])
            if P == 2 and r > 0:
                h = np.bincount(y[pi[:N0]], minlength=NT)
                ban.update(nguon_chon5=[src[int(j)] + 1 for j in top], so_lop_40_dau=int((h > 0).sum()), lop_40_dau_max=int(h.max()),
                           histogram_40_dau=[int(v) for v in h])
            out["hoan_vi"].append(ban)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "dong"])
    ap.add_argument("--T", type=float, required=True, choices=[0.6, 1.0])
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--json-b2e4r", required=True, help="JSON G0 (benchmark) hoac G1 (beta) cung T: cong hoi quy r = 0")
    ap.add_argument("--json-ktnr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ho", default="1,2")
    ap.add_argument("--r-dau", type=int, default=0)
    ap.add_argument("--r-cuoi", type=int, default=20)
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
        DG.dang_ky(B, GD, ds); assert GD.NT == DG.SO_LOP[ds]; nl = DG.SO_LUOT[ds]
    else:
        DL.dang_ky(B, GD, ds); assert GD.NT == DL.SO_LOP[ds]; nl = DL.SO_LUOT[ds]
    c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    nb = nl // GD.NT
    gb = json.load(open(a.json_b2e4r)); gk = json.load(open(a.json_ktnr))
    assert gb["ds"] == ds and abs(gb["T"] - a.T) < 1e-9 and gb["trongso"] == a.trongso and gb.get("du_fold") is True, "JSON b2e4r khac bo, cua so, trong so"
    assert gk["ds"] == ds and abs(gk["T"] - a.T) < 1e-9 and gk["alpha"] == ALPHA
    assert [p["fold"] for p in gb["tung_nguoi"]] == list(range(1, nsub + 1)) == [p["fold"] for p in gk["tung_nguoi"]]
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))
    assert cfg["nch"] == NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg["script_sha256"] == gk["trongso_script_sha256"]
    folds = list(range(nsub)) if a.folds == "all" else list(range(int(a.folds.split("-")[0]) - 1, int(a.folds.split("-")[1])))
    ho = [int(v) for v in a.ho.split(",")]; rs = list(range(a.r_dau, a.r_cuoi + 1))
    D = {"X": [], "y": []}; freqs = None
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and np.array_equal(y, np.tile(np.arange(GD.NT), nb)), "thu tu luot khong phai khoi roi muc tieu"
        freqs = np.asarray(fr, float) if freqs is None else freqs
        D["X"].append(GD.tien_xu_ly(X, ds, "causal", ns, fs)); D["y"].append(y)
    assert np.allclose(freqs, gk["freqs"])
    print("thiet bi:", dev, "| numpy", np.__version__, "| ho", ho, "| r", rs[0], "..", rs[-1], flush=True)
    t0 = time.time(); per = []
    for i in folds:
        per.append(chay_fold(i, D, a, dev, freqs, fs, GD.NT, gb["tung_nguoi"][i], gk["tung_nguoi"][i], ho, rs, nb))
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"fold {i+1} chay thong {time.time()-t0:.0f} s", flush=True)
    js = dict(seed=SEED, ds=ds, T=a.T, n0=N0, alpha=ALPHA, thiet_bi=str(dev), numpy=np.__version__, torch=torch.__version__, ho=ho, r=[rs[0], rs[-1]],
              script_sha256=sha(os.path.abspath(__file__)), b2e4r_sha256=sha(E.__file__), b1gop_sha256=sha(G.__file__),
              b1moi_lai_sha256=sha(L.__file__), global_dnn_sha256=sha(GD.__file__), b1_diem_sha256=sha(B.__file__), du_lieu_lee_sha256=sha(DL.__file__),
              json_b2e4r=a.json_b2e4r, json_b2e4r_sha256=sha(a.json_b2e4r), json_ktnr_sha256=sha(a.json_ktnr),
              trongso=a.trongso, trongso_fold_sha256=[sha(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")) for i in folds],
              du_fold=(len(folds) == nsub), tung_nguoi=per)
    json.dump(js, open(a.out + ".tmp", "w"))
    os.link(a.out + ".tmp", a.out); os.remove(a.out + ".tmp")
    if os.path.exists(a.out + ".dang-chay.tmp"):
        os.remove(a.out + ".dang-chay.tmp")
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
