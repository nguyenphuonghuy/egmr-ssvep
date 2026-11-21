#!/usr/bin/env python
"""Loi DH-TT (Duan va Hao 2026, doi:10.3390/app16083799). Chi chua ham thuan, dung chung cho huan luyen W0^EA, phat lai truc tuyen, do ZCU104 va kiem cai dat.

Dac ta:
  q = softmax(z / TAU), TAU = 2 (pt. 3).
  EA nguon: R_train = TB cov tung luot nguon (tru TB tung kenh, mau so T-1; pt. 6-8); x~ = R^{-1/2} x (pt. 11).
  EA truc tuyen: R(0) = R_train, n(0) = N0_EA; R(i) = (n R + OMEGA cov(x_i)) / (n + OMEGA); n += OMEGA; x~_i = R(i)^{-1/2} x_i (pt. 12-15).
  Mat mat: L = H(q) + LAMBDA * R_freq, R_freq = sum q~ log q~, q~_k = q_k (BETA + F_k) / sum_j q_j (BETA + F_j) (PHEP NHAN, pt. 19-20, Phu luc A).
  F^(0) = 0; F giu co dinh khi lay gradient; sau luot: F_k += [argmax q = k va q_k > GAMMA] (dau > chat).
  Cap nhat: SGD thuan theta -= ALPHA grad (pt. 5), mot buoc moi luot.
Hiep phuong sai tinh tren cua so phan tich cua dai loc thu nhat; R^{-1/2} nhan vao ca 3 dai da loc (loc mot lan);
  R^{-1/2} bang eigh tren ma tran doi xung hoa, kep tri rieng >= 1e-6 trace / C.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

TAU, N0_EA, OMEGA, GAMMA, BETA, LAMBDA = 2.0, 100.0, 0.7, 0.5, 40.0, 1.1
ALPHA_BAI = 1e-5
ALPHAS = (1e-5, 1e-4, 1e-3)
KEP = 1e-6


def cov_luot(x):
    """x (C, T) float64 -> (C, C): tru TB tung kenh, mau so T - 1 (pt. 6-7)."""
    x = np.asarray(x, np.float64)
    xc = x - x.mean(1, keepdims=True)
    return xc @ xc.T / (x.shape[1] - 1)


def nghich_can(R):
    """R^{-1/2} cua ma tran doi xung xac dinh duong; doi xung hoa, kep tri rieng >= KEP * trace / C."""
    R = 0.5 * (np.asarray(R, np.float64) + np.asarray(R, np.float64).T)
    C = R.shape[0]
    lam, V = np.linalg.eigh(R)
    san = KEP * max(float(np.trace(R)), 0.0) / C
    lam = np.maximum(lam, san if san > 0 else KEP)
    return (V * lam ** -0.5) @ V.T


def r_nguon(Xs):
    """Xs (n, NSB, C, ns): cov tren dai 1 cua moi luot nguon, lay TB (pt. 8)."""
    return np.mean([cov_luot(x[0]) for x in Xs], axis=0)


def can_chinh(X, P):
    """X (..., NSB, C, ns), P (C, C) -> P x tren moi dai (cung mot ma tran cho 3 dai)."""
    return np.einsum("cd,...rdt->...rct", P, X).astype(np.float32)


class EATrucTuyen:
    """Trung binh tich luy co trong so nguon (pt. 12-15). cap_nhat(x) cap nhat R bang luot hien tai roi tra ma tran can chinh."""

    def __init__(self, R_train, n0=N0_EA, omega=OMEGA):
        self.R = np.array(R_train, np.float64); self.n = float(n0); self.omega = float(omega)

    def cap_nhat_tu_cov(self, Cv):
        self.R = (self.n * self.R + self.omega * np.asarray(Cv, np.float64)) / (self.n + self.omega)
        self.n += self.omega
        return nghich_can(self.R)

    def cap_nhat(self, x_dai1):
        return self.cap_nhat_tu_cov(cov_luot(x_dai1))


def mat_mat(z, F, tau=TAU, beta=BETA, lam=LAMBDA):
    """z (1, K) logit, F (K,) bo dem lich su (khong lay gradient). Tra (L, logq) voi logq = log_softmax(z / tau)."""
    logq = Fn.log_softmax(z / tau, dim=-1)
    q = logq.exp()
    H = -(q * logq).sum(-1)
    logw = torch.log(beta + F.to(z.dtype)).detach()
    lqt = logq + logw
    lqt = lqt - torch.logsumexp(lqt, dim=-1, keepdim=True)                # log q~ (phep nhan, chuan hoa day du)
    Rf = (lqt.exp() * lqt).sum(-1)
    return (H + lam * Rf).sum(), logq


def cap_nhat_dem(F, logq, gamma=GAMMA):
    """F += [argmax q = k va q_k > gamma] (pt. 17-18), dung q TRUOC cap nhat (cung q voi quyet dinh)."""
    q = logq.detach().exp()[0]
    k = int(q.argmax())
    if float(q[k]) > gamma:
        F[k] += 1
    return F


def phat_lai_dh(model0, X, alpha, dev, R_train=None, t0=0, ghi_vet=False):
    """Phat lai nhan qua mot nguoi. model0 khong bi sua (sao sau). X (n, NSB, C, ns) float32 DA LOC, CHUA can chinh.
    R_train None: khong EA (nhanh C). alpha = 0: khong cap nhat tham so (A0, C alpha 0) nhung van cap nhat EA va F.
    Luot t < t0 khong xu ly (nhanh C bat dau o luot 41 = chi so 40). Quyet dinh = argmax z tinh TRUOC cap nhat.
    Tra dict: dec (n,; -1 khi t < t0), logit (n, K) float64 truoc cap nhat, F cuoi, so luot q_max > gamma, chuan gradient, so bo."""
    import copy
    model = copy.deepcopy(model0).to(dev); model.eval()                    # eval xuyen suot: tat dropout
    ps = list(model.parameters())
    for p in ps:
        p.requires_grad_(alpha > 0)
    n = X.shape[0]; K = model.fc.out_features
    F = torch.zeros(K, dtype=torch.float64)
    ea = EATrucTuyen(R_train) if R_train is not None else None
    dec = np.full(n, -1, np.int64); lg = np.full((n, K), np.nan); chuan = []; bo = 0; vuot = 0
    vet = dict(R=[], F=[], theta=[]) if ghi_vet else None
    for t in range(t0, n):
        x = X[t]
        if ea is not None:
            x = can_chinh(x, ea.cap_nhat(x[0]))
        xt = torch.tensor(np.ascontiguousarray(x[None]), device=dev, dtype=torch.float32)
        for p in ps:
            p.grad = None
        with torch.set_grad_enabled(alpha > 0):
            z = model(xt)
            L, logq = mat_mat(z, F.to(dev))
        if not (torch.isfinite(z).all() and torch.isfinite(L)):
            raise FloatingPointError(f"logit hoac mat mat khong huu han o luot {t + 1}: dung, khong cham quyet dinh")
        dec[t] = int(z.detach().argmax(1)[0]); lg[t] = z.detach().double().cpu().numpy()[0]
        if alpha > 0:
            L.backward()
            b, c = buoc_sgd(ps, alpha); bo += b
            if not b:
                chuan.append(c)
        qm = float(logq.detach().exp().max())
        vuot += int(qm > GAMMA)
        F = cap_nhat_dem(F, logq.cpu())
        if ghi_vet:
            vet["R"].append(None if ea is None else ea.R.copy()); vet["F"].append(F.clone().numpy())
            vet["theta"].append(np.concatenate([p.detach().double().cpu().numpy().ravel() for p in ps]))
    return dict(dec=dec, logit=lg, F=F.numpy(), vuot_gamma=vuot, so_luot=int(n - t0), chuan_tb=(float(np.mean(chuan)) if chuan else 0.0),
                bo_khong_huu_han=bo, model=model, vet=vet)


def buoc_sgd(params, alpha):
    """SGD thuan, khong quan tinh, khong suy giam trong so. Tra (so_bo_khong_huu_han, chuan_gradient)."""
    gs = [p.grad for p in params]
    if any(g is None for g in gs):
        raise RuntimeError("thieu gradient")
    if not all(torch.isfinite(g).all() for g in gs):
        return 1, float("nan")
    chuan = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs)))
    with torch.no_grad():
        for p in params:
            p -= alpha * p.grad
    return 0, chuan
