#!/usr/bin/env python
"""Kiem cai dat b2e4r_entropy.py tren mang nho ngau nhien, CPU, float64. In tung phep kiem, thoat 1 neu co phep khong dat.
1. Gradient theo lam cua b, Hs, Hz, Hl (qua grad_lam) so voi sai phan trung tam float64 (h 1e-5, sai so tuong doi < 1e-4), z(R) giu hang.
2. Entropy cua phan phoi deu = ln 40; entropy huu han voi logit +-1e4; gradient entropy theo u bang 0 khi u deu.
3. phat_lai_tq('b') trung b2e4r.phat_lai goc (day_gop True) tung quyet dinh, lam cuoi, G.
4. Nhan mat mat voi hang so 3: quy dao lam khong doi (Hs, Hl).
5. Moi nhanh: luot 41 cung quyet dinh; theta, G rieng (Hs khong bi anh huong boi gradient chan doan: chan_doan True/False cho cung quy dao).
"""
import math, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2e4r_entropy as E     # noqa: E402  (boc B2.phat_lai khi import; kiem dung ham goc E._phat_lai_goc)
B2 = E.B2
torch.set_default_dtype(torch.float64)
dat = []


def kiem(ten, ok, ghi=""):
    dat.append(bool(ok)); print(("DAT  " if ok else "KHONG") + " " + ten + ("  " + ghi if ghi else ""), flush=True)


torch.manual_seed(1); rng = np.random.default_rng(1)
C, ns, n, K = 9, 50, B2.N0 + 30, 5
model = B2.GD.GlobalDNN(C, ns, nch=B2.NCH, p_cuoi=0.95).double().eval()
for m_ in (model.l2, model.l3, model.l4, model.fc):
    torch.nn.init.normal_(m_.weight, 0, 0.3)
for p in model.parameters():
    p.requires_grad_(False)
X = torch.randn(n, 3, C, ns)
h1 = B2.L.truoc(model, X)
DW = 0.3 * torch.randn(K, B2.NCH, 1, C, 1); DB = 0.3 * torch.randn(K, B2.NCH)
zR = B2.z(rng.standard_normal((n - B2.N0, 40)))
lg_c = B2.L.sau(model, h1, B2.L.mod_chung("B", C, "cpu"))

# 1. gradient
for ten in E.NHANH:
    loi = 0.0
    for t in (B2.N0, B2.N0 + 7):
        lam0 = torch.softmax(torch.randn(K), 0)
        zr = torch.tensor(zR[t - B2.N0:t - B2.N0 + 1])
        def f(l):
            return float(E.mat_mat(ten, B2.L.sau(model, h1[t:t + 1], B2.G.gop(DW, DB, l)), zr))
        th = torch.log(lam0)
        _, g = E.grad_lam(model, h1[t:t + 1], DW, DB, th, ten, zr)
        sd = torch.zeros(K)
        for k in range(K):
            e = torch.zeros(K); e[k] = 1e-5
            sd[k] = (f(lam0 + e) - f(lam0 - e)) / 2e-5
        if not (torch.isfinite(g).all() and torch.isfinite(sd).all()):
            loi = float("nan"); break
        e_ = float((g - sd).abs().max() / max(float(sd.abs().max()), 1e-8))
        loi = e_ if not math.isfinite(e_) else max(loi, e_)
        if not math.isfinite(loi):
            break
    kiem(f"gradient theo lam, {ten}", math.isfinite(loi) and loi < 1e-4, f"sai so tuong doi {loi:.2e}")
kiem("phep gom sai so bat duoc NaN (tu kiem)", not (math.isfinite(float('nan')) and float('nan') < 1e-4))

# 2. entropy
kiem("H(deu) = ln 40", abs(float(E.entropy(torch.zeros(1, 40))) - math.log(40)) < 1e-12)
u = torch.tensor([[1e4, -1e4] + [0.0] * 38]); kiem("H huu han voi logit +-1e4", bool(torch.isfinite(E.entropy(u)).all()))
u0 = torch.zeros(1, 40, requires_grad=True); (gu,) = torch.autograd.grad(E.entropy(u0).sum(), u0)
kiem("dH/du = 0 khi u deu", float(gu.abs().max()) < 1e-12)

# 3. b tai lap trung phat_lai goc
dec_g, lg_g, th_g = E._phat_lai_goc(model, h1, lg_c, DW, DB, zR, B2.ALPHA, True)
lg_b, lam_b, them_b = E.phat_lai_tq(model, h1, DW, DB, zR, "b")
d_g = (B2.z(lg_g) + B2.ALPHA * zR).argmax(1); d_b = (B2.z(lg_b) + B2.ALPHA * zR).argmax(1)
kiem("b tai lap trung phat_lai goc (quyet dinh)", np.array_equal(d_g, d_b))
kiem("b tai lap trung phat_lai goc (lam cuoi, G)", np.allclose(th_g["lam"], them_b["lam"], rtol=0, atol=1e-12) and abs(th_g["G"] - them_b["G"]) < 1e-12)
kiem("lam cua b da dich khoi deu (ca kiem khong tam thuong)", them_b["tv"] > 1e-3, f"TV {them_b['tv']:.3g}")

# 4. bat bien nhan hang so
goc_mm = E.mat_mat
for ten in ("Hs", "Hl"):
    _, l1, _ = E.phat_lai_tq(model, h1, DW, DB, zR, ten)
    E.mat_mat = lambda t_, lg, zr: 3.0 * goc_mm(t_, lg, zr)
    _, l3, _ = E.phat_lai_tq(model, h1, DW, DB, zR, ten)
    E.mat_mat = goc_mm
    kiem(f"nhan mat mat voi 3 khong doi quy dao lam, {ten}", np.allclose(l1, l3, rtol=0, atol=1e-10))

# 5. luot 41, chan doan khong anh huong quy dao
ds = {}
for ten in E.NHANH:
    lg, lam_t, them = E.phat_lai_tq(model, h1, DW, DB, zR, ten, chan_doan=True)
    ds[ten] = (B2.z(lg) + B2.ALPHA * zR).argmax(1)
    kiem(f"lam_41 deu, {ten}", np.allclose(lam_t[0], 1.0 / K, rtol=0, atol=1e-12))
kiem("luot 41 moi nhanh cung quyet dinh", len({int(ds[t][0]) for t in E.NHANH}) == 1)
_, la, ta = E.phat_lai_tq(model, h1, DW, DB, zR, "Hs", chan_doan=True)
_, lb, tb = E.phat_lai_tq(model, h1, DW, DB, zR, "Hs", chan_doan=False)
kiem("gradient chan doan khong doi quy dao Hs", np.array_equal(la, lb) and ta["G"] == tb["G"])
kiem("rho tinh duoc tren Hs", ta.get("rho_tb") is not None and -1 <= ta["rho_tb"] <= 1, f"rho_tb {ta.get('rho_tb')}, bo {ta.get('rho_bo')}")
kiem("hai nhanh b va Hs khong trung quy dao (ca kiem khong tam thuong)", not np.allclose(lam_b, la, rtol=0, atol=1e-6))

print("TONG:", sum(dat), "/", len(dat))
sys.exit(0 if all(dat) else 1)
