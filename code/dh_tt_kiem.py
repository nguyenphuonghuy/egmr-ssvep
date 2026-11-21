#!/usr/bin/env python
"""Kiem cai dat DH-TT. Chay CPU, khong can du lieu that.
Thoat ma 0 khi moi kiem dat; in tung kiem. Ca so tinh tay doc lap.
"""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dh_tt_loi as D        # noqa: E402
import global_dnn as GD      # noqa: E402

torch.set_default_dtype(torch.float32)
loi = []


def kiem(ten, dk, chi_tiet=""):
    print(("DAT  " if dk else "LOI  ") + ten + (f"  ({chi_tiet})" if chi_tiet else ""), flush=True)
    if not dk:
        loi.append(ten)


# 1. EA truc tuyen, ca 3x3 duong cheo tinh tay
ea = D.EATrucTuyen(np.diag([1.0, 4.0, 9.0]))
P = ea.cap_nhat_tu_cov(np.diag([4.0, 1.0, 16.0]))
kiem("EA 3x3: n1 = 100,7", abs(ea.n - 100.7) < 1e-12)
kiem("EA 3x3: R1", np.allclose(np.diag(ea.R), [1.02085402, 3.97914598, 9.04865938], atol=1e-8), str(np.diag(ea.R)))
kiem("EA 3x3: R1^-1/2", np.allclose(np.diag(P), [0.98973329, 0.50130849, 0.33243587], atol=1e-8) and np.allclose(P, np.diag(np.diag(P))))

# 2. EA ngoai duong cheo: so voi phan ra truc tiep; P R P = I
rng = np.random.default_rng(0)
A = rng.standard_normal((5, 5)); R = A @ A.T + 0.5 * np.eye(5)
P = D.nghich_can(R)
kiem("R^-1/2 ngoai duong cheo: P R P = I", np.allclose(P @ R @ P, np.eye(5), atol=1e-9))
# ma tran suy bien (hai kenh trung): huu han, doi xung
x = rng.standard_normal((4, 300)); x[3] = x[2]
P = D.nghich_can(D.cov_luot(x))
kiem("R suy bien (kenh trung): huu han", np.isfinite(P).all() and np.allclose(P, P.T))
# cov: mau so T-1, tru TB
kiem("cov_luot khop np.cov", np.allclose(D.cov_luot(x), np.cov(x)))

# 3. R_freq voi F khong deu (bat loi chia thay vi nhan)
q = torch.tensor([[0.6, 0.2, 0.1, 0.1]], dtype=torch.float64)
z = (D.TAU * torch.log(q)).clone()
F = torch.tensor([80.0, 0.0, 0.0, 0.0], dtype=torch.float64)
L, logq = D.mat_mat(z, F)
H = float(-(q * q.log()).sum())
Rf = (float(L) - H) / D.LAMBDA
kiem("R_freq ca tinh tay = -0,663179", abs(Rf - (-0.663179)) < 5e-6, f"{Rf:.6f}")
kiem("L ca tinh tay = +0,359403", abs(float(L) - 0.359403) < 5e-6, f"{float(L):.6f}")
zz = z.clone().requires_grad_(True)
Lg, _ = D.mat_mat(zz, F); (g,) = torch.autograd.grad(Lg, zz)
kiem("dL/dz1 ca tinh tay = +0,034707 (tau 2)", abs(float(g[0, 0]) - 0.034707) < 5e-6, f"{float(g[0, 0]):.6f}")

# 4. F deu: L = (1 - lambda) H = -0,1 H
L0, _ = D.mat_mat(z, torch.zeros(4, dtype=torch.float64))
kiem("F = 0: L = -0,1 H", abs(float(L0) - (1 - D.LAMBDA) * H) < 1e-12)

# 5. Dem nhan gia: q_max dung bang gamma khong dem (dau > chat); tau = 2 dung trong q
F2 = torch.zeros(2, dtype=torch.float64)
_, lq = D.mat_mat(torch.zeros(1, 2, dtype=torch.float64), F2)
kiem("q_max = gamma = 0,5: khong dem", float(D.cap_nhat_dem(F2.clone(), lq).sum()) == 0)
zt = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
_, lq = D.mat_mat(zt, torch.zeros(4, dtype=torch.float64))
kiem("tau = 2: q_max(2,0,0,0) = 0,475367 < gamma: khong dem",
     abs(float(lq.exp().max()) - 0.475367) < 5e-6 and float(D.cap_nhat_dem(torch.zeros(4, dtype=torch.float64), lq).sum()) == 0)

# 6. Gradient theo tham so mang bang sai phan huu han; buoc SGD
torch.manual_seed(1)
m = GD.GlobalDNN(9, 50, nclass=40, nch=4).double().eval()
with torch.no_grad():                                                    # trong so lon hon de gradient du lon so voi sai so lam tron
    for p_ in m.parameters():
        p_.normal_(0.0, 0.3)
xin = torch.randn(1, 3, 9, 50, dtype=torch.float64)
Fk = torch.tensor(rng.integers(0, 5, 40), dtype=torch.float64)
L1, _ = D.mat_mat(m(xin), Fk); L1.backward()
for ten_p in ("l1.weight", "l2.weight", "l4.weight", "fc.weight", "fc.bias"):
    p = dict(m.named_parameters())[ten_p]
    huong = p.grad / p.grad.norm()                                       # huong gradient: dao ham co huong = |grad|, du lon de so
    eps = 1e-5
    with torch.no_grad():
        p += eps * huong; Lp = float(D.mat_mat(m(xin), Fk)[0]); p -= 2 * eps * huong; Lm = float(D.mat_mat(m(xin), Fk)[0]); p += eps * huong
    sp = (Lp - Lm) / (2 * eps); ad = float((p.grad * huong).sum())
    kiem(f"gradient = sai phan huu han ({ten_p})", ad > 0 and abs(sp - ad) <= 1e-5 * ad, f"{sp:.6e} vs {ad:.6e}")
truoc = [q_.detach().clone() for q_ in m.parameters()]
b, _ = D.buoc_sgd(list(m.parameters()), 1e-3)
kiem("buoc SGD: theta - alpha grad", b == 0 and all(torch.allclose(q_, t_ - 1e-3 * q_.grad) for q_, t_ in zip(m.parameters(), truoc)))

# 7. Phat lai: alpha 0 trung A0 (tham so khong doi); bat bien tien to; khong dung nhan
torch.manual_seed(2)
m0 = GD.GlobalDNN(9, 50, nclass=40, nch=4).eval()
X = rng.standard_normal((60, 3, 9, 50)).astype(np.float32)
Rt = D.r_nguon(rng.standard_normal((30, 3, 9, 50)))
goc = [p_.detach().clone() for p_ in m0.parameters()]
a0 = D.phat_lai_dh(m0, X, 0.0, "cpu", Rt, ghi_vet=True)
kiem("alpha 0: tham so khong doi", all(torch.equal(p1, p2) for p1, p2 in zip(a0["model"].parameters(), m0.parameters())))
a1 = D.phat_lai_dh(m0, X, 1e-3, "cpu", Rt, ghi_vet=True)
kiem("model goc khong bi sua sau phat lai", all(torch.equal(p1, p2) for p1, p2 in zip(m0.parameters(), goc)))
kiem("alpha 1e-3: tham so co doi", any(not torch.equal(p1, p2) for p1, p2 in zip(a1["model"].parameters(), goc)))
a1p = D.phat_lai_dh(m0, X[:25], 1e-3, "cpu", Rt, ghi_vet=True)
kiem("bat bien tien to: quyet dinh", np.array_equal(a1p["dec"], a1["dec"][:25]))
kiem("bat bien tien to: quy dao R va F", all(np.allclose(r1, r2) for r1, r2 in zip(a1p["vet"]["R"], a1["vet"]["R"][:25]))
     and all(np.array_equal(f1, f2) for f1, f2 in zip(a1p["vet"]["F"], a1["vet"]["F"][:25])))
kiem("bat bien tien to: quy dao theta", all(np.array_equal(t1, t2) for t1, t2 in zip(a1p["vet"]["theta"], a1["vet"]["theta"][:25])))
Xnan = X.copy(); Xnan[30, 0, 0, 0] = np.nan
try:
    D.phat_lai_dh(m0, Xnan, 1e-3, "cpu", None); kiem("dau vao NaN: dung", False)
except FloatingPointError:
    kiem("dau vao NaN: dung, khong cham", True)
kiem("quy dao R cua alpha 0 trung alpha > 0 (EA khong phu thuoc tham so)", all(np.allclose(r1, r2) for r1, r2 in zip(a0["vet"]["R"], a1["vet"]["R"])))
c0 = D.phat_lai_dh(m0, X, 0.0, "cpu", None, t0=40)
kiem("t0 = 40: luot truoc khong xu ly", (c0["dec"][:40] == -1).all() and (c0["dec"][40:] >= 0).all())
kiem("khong EA, alpha 0: quyet dinh = argmax mo hinh", np.array_equal(c0["dec"][40:], m0(torch.tensor(X[40:])).argmax(1).numpy()))

print("\nTONG:", "DAT" if not loi else f"{len(loi)} LOI: {loi}")
sys.exit(1 if loi else 0)
