#!/usr/bin/env python
"""B2 phu luc 2: doi chung mat mat entropy tren cung khong gian 5 he so.

Cach lam: GOI LAI NGUYEN b2e4r.py co san tren may (eb6f7225... cho Benchmark/BETA, e56f4877... cho Dong 2023; ca hai da sinh JSON G0/G1/X1),
giu moi cong cua no (nguon chon, E0, E4, z_ktn, z_R, KTNR, anh xa luot). Chi boc ham b2e4r.phat_lai: o loi goi day_gop = True (nhanh b),
cung dau vao (mang, h1, logit chung, DW, DB, zR) duoc phat lai them bang ham tong quat phat_lai_tq voi bon mat mat:
  b  : -c(s)               (tai lap nhanh b; phai trung tung quyet dinh cua b2e4r va dec.E4Rb trong JSON da co)
  Hs : H(softmax(s))       s = z(logit(lam)) + ALPHA z(R_t), z(R_t) hang so theo lam
  Hz : H(softmax(z(logit)))
  Hl : H(softmax(logit))   logit tho
H(u) = -sum softmax(u) * log_softmax(u), log tu nhien, nhiet do 1. Moi nhanh co theta, G, bo dem rieng, theta_41 = 0. Gradient theo lam
(khong theo theta), theta <- theta - eta g, eta = sqrt(2 ln K / (G^2 T)), G = max ||g||_inf, bo luot gradient khong huu han (dem),
quyet dinh truoc cap nhat. Quyet dinh moi nhanh = argmax[z(logit) + ALPHA z(R)] bang numpy float64 nhu d_b cua b2e4r.
Chan doan (chi bao): rho_t (cosin g~ cua -c(s) va H(softmax(s)) tai lam_t cua Hs, hai do thi rieng, g~ = g - TB(g)), cosin tren g tho,
r_t = ||g||_inf / G, ||eta g~||_inf, TV lam cuoi, L1(lam_b,t - lam_Hs,t), entropy va bien logit cua Hl.
Cong (lech thi dung): b tai lap trung quyet dinh b2e4r tung luot; dec.E4Rb, dec.E4Ra, dec.KTNR trung JSON da co (--json-b2e4r, kiem SHA,
ds, T, fold); lam_b, G_b trung JSON toi 1e-6; luot 41 moi nhanh trung b.
Chi ho tro goi CLI mot lan moi tien trinh (boc ham toan cuc cua b2e4r). Dung: python b2e4r_entropy.py --json-b2e4r <JSON G0/G1/X1 cung bo, cung T> --out-entropy <json ra> <doi so cua b2e4r.py, --out la tep tam>
"""
import hashlib, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2e4r as B2           # noqa: E402

B2_SHA_HOP_LE = {"eb6f7225ee62093865f13c9cf5caa0ebe4dc00e092f44e3705be69abf6962dad",      # Benchmark, BETA (G0, G1)
                 "e56f4877ff9e0b36b430e441ce36ce2d37abf9d78d7f46f6bd26d8daffc918b9"}      # Dong 2023 (X1)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
N0, ALPHA = B2.N0, B2.ALPHA
NHANH = ("b", "Hs", "Hz", "Hl")


def entropy(u):
    lp = torch.log_softmax(u, dim=-1)
    return -(lp.exp() * lp).sum(-1)


def mat_mat(ten, lg, zr):
    """lg (1, N) logit tai lam; zr (1, N) z(R_t) hang so. Tra mat mat vo huong."""
    if ten == "b":
        return -B2.L.do_tin(B2.z_t(lg) + ALPHA * zr).sum()
    if ten == "Hs":
        return entropy(B2.z_t(lg) + ALPHA * zr).sum()
    if ten == "Hz":
        return entropy(B2.z_t(lg)).sum()
    if ten == "Hl":
        return entropy(lg).sum()
    raise ValueError(ten)


def grad_lam(model, h1t, DW, DB, theta, ten, zr):
    lv = torch.softmax(theta, 0).requires_grad_(True)
    lg = B2.L.sau(model, h1t, B2.G.gop(DW, DB, lv))
    (g,) = torch.autograd.grad(mat_mat(ten, lg, zr), lv)
    return lg.detach(), g


def phat_lai_tq(model, h1, DW, DB, zR, ten, chan_doan=False):
    """Nhu b2e4r.phat_lai nhung mat mat theo ten. Tra logit truoc cap nhat (T, N) float64, lam tung luot (T, K), them."""
    K = DW.shape[0]; n = h1.shape[0]; dev = h1.device; T = n - N0
    theta = torch.zeros(K, device=dev); Gm = 0.0; eta = 0.0; bo = 0; bo_G0 = 0
    zRt = torch.tensor(zR, device=dev, dtype=torch.float32)
    lgs, lams, r_, buoc, rho, rho_tho, Hl_H, Hl_bien = [], [], [], [], [], [], [], []
    for t in range(N0, n):
        zr = zRt[t - N0:t - N0 + 1]
        lams.append(torch.softmax(theta, 0).detach().cpu().numpy().astype(np.float64))
        lg, g = grad_lam(model, h1[t:t + 1], DW, DB, theta, ten, zr)
        lgs.append(lg.double().cpu().numpy()[0])
        if chan_doan and ten == "Hs":
            _, gc = grad_lam(model, h1[t:t + 1], DW, DB, theta, "b", zr)          # do thi rieng; khong cap nhat theta, G cua Hs
            a_, b_ = g.double(), gc.double()
            if torch.isfinite(a_).all() and torch.isfinite(b_).all():
                at, bt = a_ - a_.mean(), b_ - b_.mean()
                if float(at.norm()) > 1e-12 and float(bt.norm()) > 1e-12:
                    rho.append(float((at @ bt) / (at.norm() * bt.norm())))
                if float(a_.norm()) > 1e-12 and float(b_.norm()) > 1e-12:
                    rho_tho.append(float((a_ @ b_) / (a_.norm() * b_.norm())))
        if chan_doan and ten == "Hl":
            Hl_H.append(float(entropy(lg)[0])); s2 = torch.sort(lg[0], descending=True)[0]; Hl_bien.append(float(s2[0] - s2[1]))
        if not torch.isfinite(g).all():
            bo += 1; continue
        gi = float(g.abs().max()); Gm = max(Gm, gi)
        if Gm > 0:
            eta = math.sqrt(2.0 * math.log(K) / (Gm * Gm * T))
            r_.append(gi / Gm); buoc.append(float((eta * (g - g.mean())).abs().max()))
            theta = theta - eta * g
        else:
            bo_G0 += 1
    lam = torch.softmax(theta, 0).cpu().numpy()
    them = dict(lam=lam.tolist(), tv=B2.G.tv(lam), G=Gm, eta_cuoi=eta, bo_khong_huu_han=bo, bo_G0=bo_G0,
                r_tb=(float(np.mean(r_)) if r_ else None), buoc_tb=(float(np.mean(buoc)) if buoc else None))
    if chan_doan and ten == "Hs":
        them.update(rho_tb=(float(np.mean(rho)) if rho else None), rho_so_luot=len(rho), rho_bo=T - len(rho),
                    rho_tho_tb=(float(np.mean(rho_tho)) if rho_tho else None))
    if chan_doan and ten == "Hl":
        them.update(H_trung_vi=float(np.median(Hl_H)), H_tb=float(np.mean(Hl_H)), bien_trung_vi=float(np.median(Hl_bien)),
                    H_luot=[round(v, 6) for v in Hl_H], bien_luot=[round(v, 6) for v in Hl_bien])          # phan bo theo luot
    return np.array(lgs), np.array(lams), them


KHO = {}
_phat_lai_goc = B2.phat_lai


def phat_lai_boc(model, h1, lg_c, DW, DB, zR, alpha, day_gop):
    out = _phat_lai_goc(model, h1, lg_c, DW, DB, zR, alpha, day_gop)
    if day_gop and alpha == ALPHA:
        assert "dang" not in KHO, "phat_lai nhanh b goi hai lan trong mot fold"
        dec_b, lg_b, th_b = out
        d_goc = (B2.z(lg_b) + ALPHA * zR).argmax(1)
        kq = {}
        for ten in NHANH:
            lg, lam_t, them = phat_lai_tq(model, h1, DW, DB, zR, ten, chan_doan=True)
            assert np.isfinite(lg).all(), f"logit khong huu han o nhanh {ten}"
            kq[ten] = dict(d=(B2.z(lg) + ALPHA * zR).argmax(1), lam_t=lam_t, them=them)
        assert np.array_equal(kq["b"]["d"], d_goc), "nhanh b tai lap lech quyet dinh cua b2e4r.phat_lai"
        assert np.allclose(kq["b"]["them"]["lam"], th_b["lam"], rtol=0, atol=1e-6) and abs(kq["b"]["them"]["G"] - th_b["G"]) <= 1e-6 * max(1.0, th_b["G"])
        for ten in NHANH:
            assert kq[ten]["d"][0] == kq["b"]["d"][0], f"luot 41 cua {ten} khac b"
        KHO["dang"] = kq
    return out


B2.phat_lai = phat_lai_boc


def main():
    argv = sys.argv[1:]
    def lay(ten):
        i = argv.index(ten); v = argv[i + 1]; del argv[i:i + 2]; return v
    jb, ra = lay("--json-b2e4r"), lay("--out-entropy")
    assert "--kiem" not in argv, "khong dung --kiem (goi them phat_lai alpha 0)"
    assert sha(B2.__file__) in B2_SHA_HOP_LE, "b2e4r.py khong phai ban da sinh JSON G0/G1/X1"
    assert not os.path.exists(ra), "tep ket qua da co: khong ghi de"
    ref = json.load(open(jb)); ref_sha = sha(jb)
    assert ref["script_sha256"] == sha(B2.__file__), "JSON b2e4r sinh boi ban b2e4r.py khac"
    ref_fold = {p["fold"]: p for p in ref["tung_nguoi"]}
    goc = B2.chay_fold
    tong = []

    def chay_fold_boc(i, D, a, dev, freqs, fs, NT, gg, gk, kiem):
        assert a.ds == ref["ds"] and abs(a.T - ref["T"]) < 1e-9 and a.trongso == ref["trongso"], "JSON b2e4r khac bo, cua so hoac trong so"
        KHO.clear()
        res = goc(i, D, a, dev, freqs, fs, NT, gg, gk, kiem)
        kq = KHO.pop("dang")
        r0 = ref_fold[i + 1]
        for k in ("E4Rb", "E4Ra", "KTNR"):
            assert res["dec"][k] == r0["dec"][k], f"dec.{k} lech JSON b2e4r o fold {i+1}"
        assert np.allclose(res["lam_b"], r0["lam_b"], rtol=0, atol=1e-6) and abs(res["G_b"] - r0["G_b"]) <= 1e-6 * max(1.0, r0["G_b"]), f"lam_b/G_b lech o fold {i+1}"
        assert kq["b"]["d"].tolist() == r0["dec"]["E4Rb"], f"b tai lap lech dec.E4Rb o fold {i+1}"
        yt = np.asarray(D["y"][i][N0:]); dk = np.array(r0["dec"]["KTNR"])
        for ten in NHANH:                                                # luot 41 so truc tiep voi UM-R
            assert int(kq[ten]["d"][0]) == int(dk[0]), f"luot 41 cua {ten} khac UM-R o fold {i+1}"
        o = dict(fold=i + 1, seed=res["seed"], n_cham=int(len(yt)), KTNR=float((dk == yt).mean()), E4Ra=res["E4Ra"], E4Rb=res["E4Rb"],
                 P4_b=res["P4_b"], P4_a=res["P4_a"],
                 L1_lam_b_Hs_tb=float(np.abs(kq["b"]["lam_t"] - kq["Hs"]["lam_t"]).sum(1).mean()))
        for ten in ("Hs", "Hz", "Hl"):
            d = kq[ten]["d"]
            o[ten] = float((d == yt).mean()); o["P4_" + ten] = float((d != dk).mean()); o["them_" + ten] = kq[ten]["them"]
            o["dec_" + ten] = [int(v) for v in d]
        o["them_b"] = kq["b"]["them"]
        o["lam_t"] = {ten: np.round(kq[ten]["lam_t"], 7).tolist() for ten in NHANH}   # quy dao lam tung luot (truoc cap nhat)
        tong.append(o)
        return res

    B2.chay_fold = chay_fold_boc
    tam = argv[argv.index("--out") + 1]
    assert not os.path.exists(tam), "tep tam --out da co"
    assert len({os.path.abspath(x) for x in (tam, ra, jb)}) == 3 and not os.path.exists(ra + ".tmp"), "duong dan trung nhau hoac tep tam da co"
    sys.argv = [B2.__file__] + argv
    B2.main()
    js = dict(seed=B2.SEED, ds=ref["ds"], T=ref["T"], n0=N0, alpha=ALPHA, nhanh=list(NHANH),
              script_sha256=sha(os.path.abspath(__file__)), b2e4r_sha256=sha(B2.__file__), json_b2e4r=jb, json_b2e4r_sha256=ref_sha,
              out_b2e4r_tam=tam, out_b2e4r_tam_sha256=sha(tam), numpy=np.__version__, torch=torch.__version__,
              du_fold=(len(tong) == len(ref["tung_nguoi"])), tung_nguoi=tong)
    with open(ra + ".tmp", "x") as f:                                  # tep tam doc quyen
        json.dump(js, f)
    os.link(ra + ".tmp", ra); os.remove(ra + ".tmp")
    print("xong entropy", len(tong))


if __name__ == "__main__":
    main()
