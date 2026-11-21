#!/usr/bin/env python
"""Kiem lop (a) do dung ban chuyen cho doi thu. Phan OACCA: chi can numpy, scipy.
So ban vector hoa doi_thu_oacca.OACCA voi mot ban DICH TUNG DONG tu OACCA_acc_tsinghua_2021.m (vong lap theo tan so, CCA tinh doc lap bang
tri rieng cua ma tran hiep phuong sai, corrcoef cua numpy) tren du lieu tong hop: chuoi nhan (res1, res5, res6) va trang thai (Cxx, Cxy, ma tran PSF)
phai trung qua moi luot. Thoat ma 1 neu lech."""
import sys, os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doi_thu_oacca as O


def cca_doc_lap(X, Y):
    """X (n, p), Y (n, q): he so tuong quan chinh tac lon nhat va bo loc phia X, tinh qua tri rieng (doc lap voi QR + SVD)."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    Sxx, Syy, Sxy = Xc.T @ Xc, Yc.T @ Yc, Xc.T @ Yc
    M = np.linalg.solve(Sxx, Sxy) @ np.linalg.solve(Syy, Sxy.T)
    w, V = np.linalg.eig(M)
    j = int(np.argmax(w.real))
    return float(np.sqrt(max(w.real[j], 0))), V[:, j].real


def naive(sig_all, Y, thu_tu):
    """Dich tung dong tu ma MATLAB (dong 100-200)."""
    L, NBk, C, n = sig_all.shape; F, H2, _ = Y.shape
    fb = np.array([k ** -1.25 + 0.25 for k in range(1, NBk + 1)])
    covar = np.zeros((NBk, C, C)); Cxx = np.zeros((NBk, C, C)); Cxy = np.zeros((NBk, C, H2))
    isC = False; isC2 = False
    OWx = [None] * NBk; OWy = [None] * NBk; proto = [None] * NBk
    ra = []
    for t in thu_tu:
        rho1 = np.zeros((NBk, F)); rho5 = np.zeros((NBk, F)); rho6 = np.zeros((NBk, F)); sfx = np.zeros((NBk, C, F)); ts = []
        for k in range(NBk):
            x = sig_all[t, k]
            x = x - x.mean(1, keepdims=True)
            x = x / x.std(1, ddof=1, keepdims=True)
            ts.append(x)
            for i in range(F):
                ref = Y[i]
                r, A1 = cca_doc_lap(x.T, ref.T)
                sfx[k, :, i] = A1
                r2 = np.corrcoef(OWx[k] @ x, OWy[k] @ ref)[0, 1] if isC else 0.0
                r3 = cca_doc_lap((proto[k] @ x)[:, None], ref.T)[0] if isC2 else 0.0
                rho6[k, i] = r2 + r3 + r; rho5[k, i] = r3 + r; rho1[k, i] = r
        res1 = int(np.argmax(fb @ rho1)); res6 = int(np.argmax(fb @ rho6)); res5 = int(np.argmax(fb @ rho5))
        for k in range(NBk):
            if res1 == res6:
                s = sfx[k, :, res1]; s = s / np.linalg.norm(s)
                covar[k] += np.outer(s, s)
                proto[k] = eigh(covar[k])[1][:, -1]
            Cxx[k] += ts[k] @ ts[k].T
            Cxy[k] += ts[k] @ Y[res5].T
            A = np.block([[np.zeros((C, C)), Cxy[k]], [Cxy[k].T, np.zeros((H2, H2))]])
            Bm = np.block([[Cxx[k], np.zeros((C, H2))], [np.zeros((H2, C)), np.eye(H2)]])
            ve = eigh(A, Bm)[1][:, -1]
            OWx[k] = ve[:C]; OWy[k] = ve[C:]
        if res1 == res6:
            isC2 = True
        isC = True
        ra.append((res1, res5, res6))
        yield (res1, res5, res6), Cxx.copy(), Cxy.copy(), covar.copy()       # trang thai sau TUNG luot


def main():
    rng = np.random.default_rng(20261107)
    fs, n, C, F, L = 250, 150, 9, 8, 12
    freqs = 8 + 0.6 * np.arange(F); ph = (np.arange(F) % 4) * 0.5 * np.pi
    Y, Qy = O.tham_chieu(freqs, ph, n, fs)
    y = rng.integers(0, F, L)
    t = np.arange(n) / fs
    sig = rng.standard_normal((L, O.NB, C, n))
    mix = rng.standard_normal((F, C))
    for l in range(L):                                               # SSVEP yeu tren nen nhieu: co ca luot dung va sai
        s = np.sin(2 * np.pi * freqs[y[l]] * t + 0.7) + 0.5 * np.sin(4 * np.pi * freqs[y[l]] * t)
        sig[l] += 0.35 * mix[y[l]][None, :, None] * s
    # 1) CCA: QR + SVD so voi tri rieng
    m = O.OACCA(C, Y, Qy)
    X = sig[0, 0] - sig[0, 0].mean(1, keepdims=True); X = X / X.std(1, ddof=1, keepdims=True)
    r1, A = m.diem(X)
    loi = 0
    for i in range(F):
        r, a = cca_doc_lap(X.T, Y[i].T)
        cs = abs(a @ A[:, i]) / (np.linalg.norm(a) * np.linalg.norm(A[:, i]))
        if abs(r - r1[i]) > 1e-9 or abs(cs - 1) > 1e-7:
            print("CCA lech o tan so", i, r, r1[i], cs); loi += 1
    # 2) chuoi nhan va trang thai qua L luot
    def so_tung_luot(sig_, Y_, Qy_, ten_ca):
        # so nhan va trang thai sau TUNG luot (tung tien to), khong chi trang thai cuoi
        m_ = O.OACCA(sig_.shape[2], Y_, Qy_); ra_ = []; lech = 0.0; sai = 0
        for l, (nh, Cxx, Cxy, covar) in enumerate(naive(sig_, Y_, np.arange(sig_.shape[0]))):
            nb_ = m_.buoc(sig_[l]); ra_.append(nh)
            if nh != nb_:
                print(ten_ca, "nhan lech o luot", l, nh, nb_); sai += 1
            lech = max(lech, float(np.abs(Cxx - m_.Cxx).max()), float(np.abs(Cxy - m_.Cxy).max()), float(np.abs(covar - m_.Cpsf).max()))
        if lech > 1e-8:
            print(ten_ca, "trang thai lech", lech); sai += 1
        return ra_, lech, sai
    ra, _, sai = so_tung_luot(sig, Y, Qy, "ca 1"); loi += sai
    # ca 2: nhieu thuan, co luot res5 != res1 va res6 != res1, bao ve lua chon nhan res5 cho Cxy
    sig2 = np.random.default_rng(0).standard_normal((6, O.NB, 9, 150))
    Y2, Qy2 = O.tham_chieu(8 + 0.6 * np.arange(8), np.zeros(8), 150, 250)
    ra2, lech2, sai = so_tung_luot(sig2, Y2, Qy2, "ca 2"); loi += sai
    mong = [(4, 4, 4), (7, 2, 2), (0, 0, 5), (7, 0, 0), (4, 4, 3), (3, 3, 0)]
    if ra2 != mong:
        print("ca 2: chuoi nhan khac chuoi tinh doc lap:", ra2); loi += 1
    if not any(r[0] != r[1] for r in ra2):
        print("ca 2 khong co luot res5 != res1"); loi += 1
    print("ca 2:", ra2, "lech trang thai lon nhat %.2e" % lech2)
    khac15 = sum(r[0] != r[1] for r in ra); khac16 = sum(r[0] != r[2] for r in ra)
    print(f"OACCA: {L} luot, {F} tan so; so luot res1 != res5: {khac15}; res1 != res6: {khac16}; dung res6: {sum(r[2] == y[l] for l, r in enumerate(ra))}/{L}; loi: {loi}")
    # 3) dem MAC: in de doi chieu
    print("MAC 1,0 s:", O.dem_mac(9, 250, 40))
    sys.exit(1 if loi else 0)


if __name__ == "__main__":
    main()
