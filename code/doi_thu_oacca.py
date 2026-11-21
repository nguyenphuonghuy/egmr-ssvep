#!/usr/bin/env python
"""Doi thu OACCA (Wong va cong su, IEEE TBME 2022, DOI 10.1109/TBME.2021.3133594) cho nhanh b1gop, diem 6
Chuyen tu ma MATLAB cong khai edwin465/SSVEP-OACCA v1.2 (GPL-3.0),
ban luu o tham-chieu-matlab/OACCA_acc_tsinghua_2021.m, OACCA_acc_beta_2021.m.
Thuat toan (dong 113-200 ban Benchmark), moi luot, moi dai k (5 dai, he so k^-1,25 + 0,25):
  X = cua so, tru TB va chia do lech chuan (N-1) theo kenh; voi moi tan so i: CCA(X, Y_i) -> r1, bo loc u_i.
  r2 = corr(OWx' X, OWy' Y_i) (OMSCCA; 0 khi chua co);  r3 = CCA(w_psf' X, Y_i) (PSF; 0 khi chua co).
  nhan: res1 = argmax(sum_k r1), res5 = argmax(sum_k (r1 + r3)), res6 = argmax(sum_k (r1 + r2 + r3)) = QUYET DINH OACCA.
  cap nhat SAU quyet dinh: PSF: neu res1 == res6 thi cong tich ngoai cua u_res1 (chuan hoa) va lay vec to rieng lon nhat;
                           OMSCCA: Cxx += X X', Cxy += X Y_res5'; tri rieng tong quat [0 Cxy; Cyx 0] v = l [Cxx 0; 0 I] v, lay vec to lon nhat.
Che do:
  --che-do goc : giao thuc cua bai (lop (b)): chan luoc 50 Hz + Chebyshev I, loc hai chieu tren doan 2 s, do tre 0,12 s (Benchmark) / 0,13 s (BETA),
                 cua so 0,6..1,5 s, 10 lan xao thu tu luot, cham moi luot.
  --che-do nq  : giao thuc du an (lop (c)): tru TB doan truoc kich thich, loc NHAN QUA tu mau 0 cua doan ghi, do tre du an, thu tu tep,
                 cua so 0,6 va 1,0 s, cham luot 41..het (chinh) va ca phien.
Khac ma goc (ghi ro): loc hai chieu dung sosfiltfilt (MATLAB filtfilt ba; khac xu ly bien); CCA tinh bang QR + SVD (canoncorr cung cach).
Khong dung du lieu nguoi nguon. Nhan that chi de cham.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.signal import cheb1ord, cheby1, iircomb, lfilter, filtfilt, sosfilt, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 20261107
NB, NH, N0 = 5, 5, 40
FB = np.array([k ** -1.25 + 0.25 for k in range(1, NB + 1)])
DO_TRE_GOC = {"benchmark": 0.12, "beta": 0.13}


def lam_tron(x):
    return int(np.floor(x + 0.5))                      # round cua MATLAB


def bo_loc(fs):
    """Chan luoc 50 Hz (Q = 35) va 5 dai Chebyshev I [8k, 90] Hz nhu ma goc (cheb1ord(Wp, Ws, 3, 40), gon 0,5 dB)."""
    bn, an = iircomb(50.0, 35.0, ftype="notch", fs=fs)
    sos = []
    for k in range(1, NB + 1):
        N, Wn = cheb1ord([8 * k / (fs / 2), 90 / (fs / 2)], [(8 * k - 2) / (fs / 2), 100 / (fs / 2)], 3, 40)
        sos.append(cheby1(N, 0.5, Wn, btype="bandpass", output="sos"))
    return (bn, an), sos


def tien_xu_ly(X, che_do, on, lat_mau, ns, fs):
    """X (luot, C, t) tho. Tra (luot, NB, C, ns)."""
    (bn, an), sos = bo_loc(fs)
    out = np.empty((X.shape[0], NB, X.shape[1], ns))
    if che_do == "goc":
        seg = X[:, :, on: on + lat_mau + ns]
        seg = filtfilt(bn, an, seg, axis=2)
        for k, so in enumerate(sos):
            out[:, k] = sosfiltfilt(so, seg, axis=2)[:, :, lat_mau: lat_mau + ns]
    else:
        st = on + lat_mau
        Xb = X[:, :, : st + ns] - X[:, :, :on].mean(axis=2, keepdims=True)
        Xb = lfilter(bn, an, Xb, axis=2)
        for k, so in enumerate(sos):
            out[:, k] = sosfilt(so, Xb, axis=2)[:, :, st: st + ns]
    return out


def tham_chieu(freqs, phases, n, fs):
    """Y (F, 2 NH, n): cos roi sin, hai 1..NH, co pha nhu ma goc. Qy (n, F * 2 NH): co so truc chuan cua Y_i da tru TB (cho CCA)."""
    t = np.arange(n) / fs; h = np.arange(1, NH + 1)[:, None]
    Y = np.stack([np.concatenate([np.cos(2 * np.pi * f * h * t + p * h), np.sin(2 * np.pi * f * h * t + p * h)]) for f, p in zip(freqs, phases)])
    Qy = np.stack([np.linalg.qr((y - y.mean(1, keepdims=True)).T)[0] for y in Y])          # (F, n, 2 NH)
    return Y, Qy


class OACCA:
    def __init__(self, C, Y, Qy):
        self.C, self.Y, self.Qy = C, Y, Qy
        self.F, self.H2, self.n = Y.shape
        self.Qy2 = Qy.transpose(1, 0, 2).reshape(self.n, self.F * self.H2)
        self.Yc = Y - Y.mean(2, keepdims=True)
        self.Cxx = np.zeros((NB, C, C)); self.Cxy = np.zeros((NB, C, self.H2)); self.Cpsf = np.zeros((NB, C, C))
        self.u = np.zeros((NB, C)); self.v = np.zeros((NB, self.H2)); self.w = np.zeros((NB, C))
        self.co_om = False; self.co_psf = False

    def diem(self, Xk):
        """Xk (C, n) da chuan hoa. Tra r1, r2, r3 (F,), A (C, F) bo loc CCA."""
        Qx, Rx = np.linalg.qr(Xk.T)
        M = (Qx.T @ self.Qy2).reshape(self.C, self.F, self.H2).transpose(1, 0, 2)
        U, S, _ = np.linalg.svd(M)
        A = solve_triangular(Rx, U[:, :, 0].T)
        return S[:, 0], A

    def buoc(self, sig):
        """sig (NB, C, n): mot luot. Tra (res1, res5, res6) va cap nhat trang thai."""
        rho1 = np.empty((NB, self.F)); rho5 = np.empty_like(rho1); rho6 = np.empty_like(rho1); Xs, As = [], []
        for k in range(NB):
            X = sig[k] - sig[k].mean(1, keepdims=True)
            sd_ = X.std(1, ddof=1, keepdims=True)
            assert (sd_ > 0).all(), "kenh hang so: du lieu loi, khong tu them dieu chuan"
            X = X / sd_
            r1, A = self.diem(X)
            r2 = np.zeros(self.F); r3 = np.zeros(self.F)
            if self.co_om:
                a = self.u[k] @ X; a = a - a.mean()
                b = np.einsum("fhn,h->fn", self.Yc, self.v[k])
                r2 = (b @ a) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))
            if self.co_psf:
                p = self.w[k] @ X; p = p - p.mean()
                r3 = np.linalg.norm(np.einsum("fnh,n->fh", self.Qy, p), axis=1) / np.linalg.norm(p)
            rho1[k], rho5[k], rho6[k] = r1, r1 + r3, r1 + r2 + r3
            Xs.append(X); As.append(A)
        res1, res5, res6 = (int(np.argmax(FB @ r)) for r in (rho1, rho5, rho6))
        for k in range(NB):
            if res1 == res6:
                s = As[k][:, res1]; s = s / np.linalg.norm(s)
                self.Cpsf[k] += np.outer(s, s)
                self.w[k] = eigh(self.Cpsf[k])[1][:, -1]
            self.Cxx[k] += Xs[k] @ Xs[k].T
            self.Cxy[k] += Xs[k] @ self.Y[res5].T
            C, H2 = self.C, self.H2
            Am = np.zeros((C + H2, C + H2)); Am[:C, C:] = self.Cxy[k]; Am[C:, :C] = self.Cxy[k].T
            Bm = np.zeros_like(Am); Bm[:C, :C] = self.Cxx[k]; Bm[C:, C:] = np.eye(H2)
            ve = eigh(Am, Bm)[1][:, -1]
            u1, v1 = ve[:C].copy(), ve[C:]
            if u1[0] == 1:                                           # nhanh du phong cua ma goc (dong 180-183)
                u1 = np.zeros(C); u1[-3:] = 1
            self.u[k], self.v[k] = u1, v1
        if res1 == res6:
            self.co_psf = True
        self.co_om = True
        return res1, res5, res6


def dem_mac(C, n, F, H2=2 * NH):
    """DU TOAN phep nhan chinh moi luot cua ban cai nay (khong gom bank loc), khong phai so MAC chinh xac cua LAPACK: QR tinh 2 n C^2, SVD va tri
    rieng tinh theo bac (kich thuoc^3); cap nhat la CAN TREN (PSF tinh nhu moi luot deu cap nhat, thuc te chi khi res1 == res6)."""
    sl = NB * (C * n                       # chuan hoa
               + 2 * n * C * C             # QR cua X
               + C * n * F * H2            # Qx' Qy
               + F * C * H2 * min(C, H2)   # SVD (bac)
               + F * C * C                 # giai tam giac
               + C * n + F * H2 * n + F * n      # r2: u'X, v'Y_i, tich vo huong
               + C * n + F * n * H2)             # r3: w'X, Qy_i' p
    cn = NB * (C * C * n + C * H2 * n + (C + H2) ** 3 + C ** 3 + C * C)
    return dict(suy_luan=int(sl), cap_nhat=int(cn), tong=int(sl + cn))


def nap(ds, s, B):
    from scipy.io import loadmat
    X, y, freqs = B.load_subject(ds, s, B.CH9)
    c = B.DS[ds]
    if ds == "dong":
        import du_lieu_dong as DG
        ph = DG.PHA_DONG_PI * np.pi
    elif ds == "benchmark":
        ph = np.asarray(loadmat(os.path.join(c["root"], "Freq_Phase.mat"))["phases"]).ravel()
    else:
        ph = np.asarray(loadmat(os.path.join(c["root"], f"S{s}.mat"), squeeze_me=True, struct_as_record=False)["data"].suppl_info.phases).ravel()
    nl = {"benchmark": 240, "beta": 160, "dong": 160}[ds]
    assert len(freqs) == 40 and len(ph) == 40 and np.isfinite(X).all() and np.isfinite(freqs).all() and np.isfinite(ph).all()
    assert X.shape[0] == nl and X.shape[1] == (8 if ds == "dong" else 9) and np.array_equal(np.bincount(y, minlength=40), np.full(40, nl // 40)), f"du lieu nguoi {s}"
    return X, y, np.asarray(freqs, float), np.asarray(ph, float)


def chay(sig, y, Y, Qy, thu_tu):
    m = OACCA(sig.shape[2], Y, Qy)
    d1 = np.empty(len(thu_tu), int); d6 = np.empty(len(thu_tu), int)
    for j, t in enumerate(thu_tu):
        d1[j], _, d6[j] = m.buoc(sig[t])
    yt = y[thu_tu]
    return d1, d6, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "dong"])
    ap.add_argument("--che-do", required=True, choices=["goc", "nq"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--nguoi", default="all", help="vi du 1-5")
    ap.add_argument("--so-lan-xao", type=int, default=10)
    a = ap.parse_args()
    assert a.so_lan_xao >= 1
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    import b1_diem as B
    if a.ds == "dong":
        assert a.che_do == "nq", "Dong 2023 chi chay che do nq"
        import du_lieu_dong as DG
        DG.dang_ky(B, None, "dong")
    for d in B.DS:
        B.DS[d]["root"] = os.environ.get("B1_ROOT_" + d.upper(), B.DS[d]["root"])
    c = B.DS[a.ds]; fs = c["fs"]; on = c["onset"]; nsub = c["n"]
    ng = list(range(1, nsub + 1)) if a.nguoi == "all" else list(range(int(a.nguoi.split("-")[0]), int(a.nguoi.split("-")[1]) + 1))
    if a.che_do == "goc":
        lat = lam_tron(DO_TRE_GOC[a.ds] * fs); wins = [int(np.floor(w * fs + 1e-9)) for w in np.arange(6, 16) / 10.0]; ns_max = 2 * fs
    else:
        lat = int(round(c["lat"] * fs)); wins = [int(round(0.6 * fs)), int(round(1.0 * fs))]; ns_max = max(wins)
    t0 = time.time(); per = []
    for s in ng:
        X, y, freqs, ph = nap(a.ds, s, B)
        sig = tien_xu_ly(X, a.che_do, on, lat, ns_max, fs)
        n_l = len(y); e = {"nguoi": s, "T": {}}
        for n in wins:
            Y, Qy = tham_chieu(freqs, ph, n, fs)
            sg = sig[:, :, :, :n]
            if a.che_do == "goc":
                cca, oa = [], []
                for cv in range(a.so_lan_xao):
                    tt = np.random.default_rng(SEED + 1000 * cv).permutation(n_l)       # cung hoan vi cho moi nguoi, nhu ma goc
                    d1, d6, yt = chay(sg, y, Y, Qy, tt)
                    cca.append(float((d1 == yt).mean())); oa.append(float((d6 == yt).mean()))
                e["T"]["%.1f" % (n / fs)] = dict(cca=cca, oacca=oa)
            else:
                d1, d6, yt = chay(sg, y, Y, Qy, np.arange(n_l))
                f = lambda d: dict(sau=float((d[N0:] == yt[N0:]).mean()), phien=float((d == yt).mean()), dec=[int(v) for v in d])
                e["T"]["%.1f" % (n / fs)] = dict(cca=f(d1), oacca=f(d6))
        per.append(e)
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"nguoi {s} xong {time.time()-t0:.0f} s", flush=True)                       # khong in do chinh xac
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    js = dict(seed=SEED, ds=a.ds, che_do=a.che_do, n0=N0, nguoi=ng, kenh=B.CH9, fs=fs, onset=on, do_tre_mau=lat, cua_so_mau=wins, so_lan_xao=a.so_lan_xao,
              mac={("%.1f" % (n / fs)): dem_mac(int(sig.shape[2]), n, 40) for n in wins}, script_sha256=sha(os.path.abspath(__file__)), b1_diem_sha256=sha(B.__file__),
              tung_nguoi=per)
    json.dump(js, open(a.out, "w"), indent=1)
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
