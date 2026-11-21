#!/usr/bin/env python
"""Do thoi gian DAU-CUOI cua cac doi thu B1 tren PS cua ZCU104, cung quy trinh voi CSS-R (zcu104_ktnr_daucuoi.py).
In MOT dong JSON moi lan goi (hoac ghi --out).
 Ban 2.
Chay trong ~/ssvep_zcu104_pkg/doi_thu (can b1_diem.py, global_dnn.py, global_dnn_thichnghi.py, rcs_tai_hien.py, ensemble_dnn.py, du_lieu_eld.py,
zcu104_ktnr_daucuoi.py, du_lieu_ben/benchmark_S{1,17,27}.npz, trongso_T10/fold017_trongso.pt, cca_net/goc + cca_net/shim_zcu104).
Dong ho moi luot: bat dau khi luot tho (9 x 1500) da o bo nho, ket thuc o nhan quyet dinh; gom tien xu ly rieng cua phuong phap
(he so bo loc thiet ke MOT lan luc khoi dong). Khong gom thu nhan, khoi tao, hoc ngoai tuyen. torch.no_grad dat NGOAI dong ho cho moi phuong phap
dung PyTorch, tru CCA-Net (ma goc universal_model.test khong dung no_grad: giu nguyen).
Phan dung tu nguoi nguon (tinh ngoai tuyen, khong tinh gio): nguoi 17 va 27 tren bo, hoac trong so ngau nhien dung kich thuoc.
--pp:
  ktnr   : moc CSS-R, duong tinh cua zcu104_ktnr_daucuoi.py.
  fbcca  : FBCCA 5 dai, 5 hai, rho binh phuong (fbcca_dung.py).
  rcs    : RCS day du, duong quyet dinh (TRCA gop lop 5 dai + FBCCA 5 dai, rho khong binh phuong; rcs_tai_hien.py). Bao rieng chi phi
           hoc lai sau moi lo 20 luot voi trang thai TICH LUY nhu rcs_tai_hien.phat_lai (12 lo tren 240 luot; nhan yeu = quyet dinh cua duong do).
  ed22, edgoc : Ensemble-DNN 34 mang nguon (22 / 120 kenh): moi mang du doan + do giong + bo phieu k dong (ensemble_dnn.py); mau lop
           chieu san ngoai tuyen (co loi cho Ensemble-DNN, nhu bo dem phep nhan).
  ccanet : quyet dinh 'final' cua CCA-Net theo ma chinh thuc commit 1660d42 che do gt (cca_net_tai_hien.py dong 175-208): filter_bank, LST,
           FB_stand, mang (khong no_grad nhu ma goc), FBTtCCA voi mau du lieu (f2), fbcca x2 (f1, f3; sklearn CCA), min-max, argmax.
           KHONG tinh e1/f4 (chi dung cho bien the bao kem CNN+FBTTCCA, khong vao 'final'). Bo loc fb_nhan_qua va FBCCA.filterbank thiet ke MOT lan.
           Dem so luot co diem khong huu han hoac hang min-max khong co mien bien thien (khong doi cong thuc).
--kiem: khong do gio; so quyet dinh cua duong tung luot voi ban theo lo tren --kiem-luot luot nguoi 1; ma thoat 5 neu khac hoac co diem hong.
"""
import argparse, contextlib, functools, hashlib, importlib, json, os, sys, time
import numpy as np
import torch
from scipy.signal import sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FS, ON, C, N = 250, 125, 9, 40
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
HONG = {"n": 0}                                                              # dem luot co diem hong (CCA-Net)


def q(v):
    v = np.asarray(v, float)
    return dict(tb_ms=1e3 * float(v.mean()), trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)),
                max_ms=1e3 * float(v.max()), n=int(len(v)), huu_han=bool(np.isfinite(v).all()))


def nap(s):
    p = os.path.join(HERE, "du_lieu_ben", "benchmark_S%d.npz" % s)
    z = np.load(p)
    X, y = z["X"], z["y"]
    assert X.shape == (240, C, X.shape[2]) and X.shape[2] >= ON + 35 + 250 and np.isfinite(X).all(), (s, X.shape)
    assert np.array_equal(y, np.tile(np.arange(N), 6)) and len(z["freqs"]) == N, s           # khoi roi lop
    return X, y, z["freqs"], z["phases"]


# ---------------- CSS-R (moc) ----------------
def lam_ktnr(T, seed):
    import global_dnn as GD
    import zcu104_ktnr_daucuoi as KD
    st = ON + int(round(0.14 * FS)); ns = int(round(T * FS))
    Xr, _, freqs, _ = nap(1)
    _, sos = GD.bo_loc(FS)
    m = GD.GlobalDNN(C, ns, N, nch=22)
    if T == 1.0:
        m.load_state_dict(torch.load(os.path.join(HERE, "trongso_T10", "fold017_trongso.pt"), map_location="cpu"))
    m.eval()
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        m.l2.weight += 0.01 * torch.randn_like(m.l2.weight); m.l2.bias += 0.01 * torch.randn_like(m.l2.bias)
    w = rng.standard_normal(C); wR = w / np.linalg.norm(w)
    Qb = KD.co_so(freqs, FS, ns)

    def quyet(xr):                                                             # duong tinh cua zcu104_ktnr_daucuoi.py (dong 65-75)
        xb = xr[:, : st + ns] - xr[:, :ON].mean(axis=1, keepdims=True)
        u = np.empty((len(sos), C, ns), np.float32)
        for r, so in enumerate(sos):
            u[r] = sosfilt(so, xb, axis=1)[:, st: st + ns]
        h1 = m.l1(torch.from_numpy(u)[None])
        lg = m.fc(m.l4(torch.relu(m.l3(m.l2(h1)))).flatten(1))
        U = h1[0, 0].double().numpy()
        yv = wR @ U; yv = yv - yv.mean()
        s_ = KD.z_np(lg.double().numpy()[0]) + 0.5 * KD.z_np(KD.diem_R1(yv, Qb))
        return int(s_.argmax())

    def lo(Xr_):                                                               # ban theo lo doc lap: GD.tien_xu_ly + GlobalDNN.forward + nhanh R
        u = torch.from_numpy(GD.tien_xu_ly(Xr_, "benchmark", "causal", ns, FS))
        lg = m(u).double().numpy()
        U = m.l1(u)[:, 0].double().numpy()                                     # (n, C, ns)
        out = []
        for i in range(len(Xr_)):
            yv = wR @ U[i]; yv = yv - yv.mean()
            out.append(int((KD.z_np(lg[i]) + 0.5 * KD.z_np(KD.diem_R1(yv, Qb))).argmax()))
        return np.asarray(out)
    return quyet, lo, None


# ---------------- FBCCA ----------------
def lam_fbcca(T, seed):
    import b1_diem as B
    st = ON + int(round(B.DS["benchmark"]["lat"] * FS)); ns = int(round(T * FS))
    _, _, freqs, _ = nap(1)
    sos = B.filterbank(FS)                                                     # thiet ke MOT lan
    Qy = B.refs(freqs, FS, ns)

    def quyet(xr):
        xb = xr[:, : st + ns] - xr[:, :ON].mean(axis=1, keepdims=True)
        sc = np.zeros(N)
        for m_, so in enumerate(sos):
            x = B.center(sosfilt(so, xb, axis=1)[:, st: st + ns])
            qx = np.linalg.qr(x.T)[0]
            r = np.linalg.svd(np.einsum("tc,ktj->kcj", qx, Qy), compute_uv=False)[:, 0]
            sc += B.FBW[m_] * r ** 2
        return int(sc.argmax())

    def lo(Xr):                                                                # ban theo lo: nguyen van fbcca_dung.py
        Xp = B.preprocess(Xr, "benchmark", sos, ns)
        out = []
        for i in range(Xp.shape[0]):
            sc = np.zeros(N)
            for m_ in range(Xp.shape[1]):
                qx = np.linalg.qr(B.center(Xp[i, m_]).T)[0]
                r = np.linalg.svd(np.einsum("tc,ktj->kcj", qx, Qy), compute_uv=False)[:, 0]
                sc += B.FBW[m_] * r ** 2
            out.append(int(sc.argmax()))
        return np.asarray(out)
    return quyet, lo, None


# ---------------- RCS day du ----------------
def lam_rcs(T, seed):
    import b1_diem as B
    import rcs_tai_hien as R
    st = ON + int(round(B.DS["benchmark"]["lat"] * FS)); ns = int(round(T * FS))
    sos = B.filterbank(FS)
    _, _, freqs, _ = nap(1)
    Qr = B.refs(freqs, FS, ns); A = R.A
    src = []
    for s in (17, 27):
        X, y, _, _ = nap(s)
        src.append((B.center(B.preprocess(X, "benchmark", sos, ns)), y))
    Drep = []
    for k in range(N):                                                         # 30 luot moi lop nhu 5 dai dien x 6 khoi (lap lai luot that cua nguoi 17, 27)
        D = np.concatenate([x[y == k] for x, y in src])
        Drep.append(D[np.arange(30) % len(D)])
    W0 = np.empty((B.NB, C, N)); M0 = np.empty((N, B.NB, C, ns))
    for k in range(N):
        W0[:, :, k] = R.trca_w(Drep[k]); M0[k] = Drep[k].mean(0)
    Tp0, Tn0 = R.chieu_mau(W0, M0)

    def tien(xr):
        xb = xr[:, : st + ns] - xr[:, :ON].mean(axis=1, keepdims=True)
        return B.center(np.stack([sosfilt(so, xb, axis=1)[:, st: st + ns] for so in sos]))   # (NB, C, ns)

    def quyet(xr):
        x = tien(xr)
        fb5 = np.zeros(N)
        for b in range(B.NB):
            qx = np.linalg.qr(x[b].T)[0]
            fb5 += A[b] * np.linalg.svd(np.einsum("tc,kth->kch", qx, Qr), compute_uv=False)[:, 0]
        r = R.diem(W0, Tp0, Tn0, x)
        return int((r.T @ A + fb5).argmax())

    def lo(Xr):
        X = B.center(B.preprocess(Xr, "benchmark", sos, ns))
        out = []
        for i in range(X.shape[0]):
            f = np.array([np.linalg.svd(np.einsum("tc,kth->kch", np.linalg.qr(X[i, b].T)[0], Qr), compute_uv=False)[:, 0] for b in range(B.NB)])
            out.append(int((R.diem(W0, Tp0, Tn0, X[i]).T @ A + f.T @ A).argmax()))
        return np.asarray(out)

    def hoc_lai_tich_luy(Xr):                                                  # phat lai tuan tu nhu rcs_tai_hien.phat_lai(NB, True, False, fb5)
        n = len(Xr); x = np.stack([tien(Xr[i]) for i in range(n)])
        fb5 = np.zeros((n, N))
        for i in range(n):
            for b in range(B.NB):
                fb5[i] += A[b] * np.linalg.svd(np.einsum("tc,kth->kch", np.linalg.qr(x[i, b].T)[0], Qr), compute_uv=False)[:, 0]
        rec_end = R.N_REC                                                      # kich ban co dinh: gioi thieu 20 luot (moc muon nhat cua ma goc); khong phai can tren chi phi
        dec = np.empty(n, int); dec[:rec_end] = fb5[:rec_end].argmax(1)       # nhan yeu giai doan gioi thieu = FBCCA
        W = W0.copy(); Mt = M0.copy(); Tp, Tn_ = Tp0, Tn0; ud = [[] for _ in range(N)]; cho = list(range(rec_end)); tg = []

        def hoc_lai(cho):
            t0 = time.perf_counter()
            doi = set()
            for j in cho:
                ud[dec[j]].append(j); doi.add(int(dec[j]))
            for k in doi:
                U = x[ud[k]]
                D = np.concatenate([R.lst(Drep[k], U.mean(0)) if len(ud[k]) >= 2 else Drep[k], U])
                W[:, :, k] = R.trca_w(D); Mt[k] = D.mean(0)
            kq = R.chieu_mau(W, Mt)
            tg.append(time.perf_counter() - t0)
            return kq
        if rec_end % R.LO == 0:
            Tp, Tn_ = hoc_lai(cho); cho = []
        for i in range(rec_end, n):
            r = R.diem(W, Tp, Tn_, x[i])
            dec[i] = int((r.T @ A + fb5[i]).argmax())                          # quyet dinh bang trang thai hien tai, TRUOC khi vao thich nghi
            cho.append(i)
            if (i + 1) % R.LO == 0:
                Tp, Tn_ = hoc_lai(cho); cho = []
        return tg
    return quyet, lo, hoc_lai_tich_luy


# ---------------- Ensemble-DNN ----------------
def lam_ed(T, seed, nch):
    import global_dnn as GD
    import ensemble_dnn as E
    ns = int(round(T * FS)); S = 34
    _, _, freqs, _ = nap(1)
    _, sos = GD.bo_loc(FS)
    st = ON + int(round(0.14 * FS))
    Qr = E.tham_chieu(freqs, FS, ns, "cpu")
    mau_src = []
    for s in (17, 27):
        X, y, _, _ = nap(s)
        mau_src.append(torch.tensor(GD.tien_xu_ly(X, "benchmark", "causal", ns, FS)))
    Xs = torch.cat(mau_src); ys = torch.tensor(np.concatenate([nap(s)[1] for s in (17, 27)]))
    torch.manual_seed(seed)
    nets, pre = [], []
    with torch.no_grad():
        for j in range(S):
            m = GD.GlobalDNN(C, ns, nch=nch, p_cuoi=0.95)
            m.eval(); nets.append(m)
            wb = m.l1.weight.reshape(-1); Wl = m.l2.weight[:, 0, :, 0]
            mau = torch.einsum("b,nbct->nct", wb, torch.stack([Xs[ys == k].mean(0) for k in range(N)]))
            zt = torch.einsum("kc,nct->nkt", Wl, mau); zt = zt - zt.mean(-1, keepdim=True)     # chieu san ngoai tuyen (N, K, ns)
            pre.append((wb, Wl, zt))

    def mot(xu):
        P, Sm = [], []
        for m, (wb, Wl, zt_all) in zip(nets, pre):
            pr = m(xu).argmax(1)
            x1 = torch.einsum("b,ibct->ict", wb, xu)
            zx = torch.einsum("kc,ict->ikt", Wl, x1); zx = zx - zx.mean(-1, keepdim=True)
            zt = zt_all[pr]
            r = (zt * zx).sum(-1) / torch.sqrt((zt * zt).sum(-1) * (zx * zx).sum(-1) + 1e-30)
            pq = torch.einsum("ikt,itq->ikq", zx, Qr[pr])
            cca2 = (pq ** 2).sum(-1) / ((zx ** 2).sum(-1) + 1e-30)
            P.append(pr); Sm.append((r ** 2 + cca2).max(1).values)
        return torch.stack(P), torch.stack(Sm)

    def quyet(xr):
        xb = xr[:, : st + ns] - xr[:, :ON].mean(axis=1, keepdims=True)
        u = np.empty((1, len(sos), C, ns), np.float32)
        for r_, so in enumerate(sos):
            u[0, r_] = sosfilt(so, xb, axis=1)[:, st: st + ns]
        P, Sm = mot(torch.from_numpy(u))
        d, _ = E.chon_dong(Sm, P)
        return int(d[0])

    def lo(Xr):                                                                # ban theo lo cua ma goc: do_giong + chon_dong tren ca tap
        Xu = torch.tensor(GD.tien_xu_ly(Xr, "benchmark", "causal", ns, FS))
        P = torch.stack([m(Xu).argmax(1) for m in nets])
        Sm = torch.stack([E.do_giong(m, Xs, ys, Xu, P[j], Qr) for j, m in enumerate(nets)])
        return E.chon_dong(Sm, P)[0].numpy()
    return quyet, lo, None


# ---------------- CCA-Net ----------------
def lam_ccanet(T, seed):
    os.environ["CN_DS"] = "1"; os.environ["CN_T"] = repr(T); os.environ["CN_CHE"] = "gt"
    G = os.path.join(HERE, "cca_net")
    sys.path[:0] = [os.path.join(G, "shim_zcu104"), os.path.join(G, "goc")]
    for l in open(os.path.join(G, "goc", "SHA256SUMS.txt")):                 # ma goc khong bi sua
        h, f = l.split(); assert sha(os.path.join(G, "goc", f.lstrip("*"))) == h, f
    from scipy import signal
    from config import config
    import function as FN
    import FBCCA as FBm
    import CCA as CCm
    import model as Mm
    rfs, Tn, Nh, NT = config.rfs, config.T, config.Nh, config.num_class
    st = ON + int(round(0.14 * rfs))
    nyq = rfs / 2
    sos3 = []                                                                  # = cca_net_tai_hien.fb_nhan_qua, thiet ke MOT lan
    for pb, sb in zip([6, 14, 22], [4, 10, 16]):
        Nn, Wn = signal.cheb1ord([pb / nyq, 90 / nyq], [sb / nyq, 100 / nyq], 3, 40)
        sos3.append(signal.cheby1(Nn, 0.5, Wn, 'bandpass', output='sos'))
    pbF = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]; sbF = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    sosF = {}

    def fb_nhan_qua(eeg):
        out = np.zeros((eeg.shape[0], eeg.shape[1], config.Nm, eeg.shape[2], Tn))
        for k in range(config.Nm):
            out[:, :, k] = signal.sosfilt(sos3[k], eeg, axis=-1)[..., -Tn:]
        return out

    def fbF(eeg, fs, idx_fb):                                                  # = cca_net_tai_hien.fbcca_filterbank_nhan_qua, he so luu lai
        if idx_fb not in sosF:
            Nn, Wn = signal.cheb1ord([pbF[idx_fb] / (fs / 2), 90 / (fs / 2)], [sbF[idx_fb] / (fs / 2), 100 / (fs / 2)], 3, 40)
            sosF[idx_fb] = signal.cheby1(Nn, 0.5, Wn, 'bandpass', output='sos')
        return signal.sosfilt(sosF[idx_fb], eeg, axis=-1)[..., -Tn:]

    def tfb(self, X):
        return np.stack([signal.sosfilt(s_, X, axis=-1)[..., -Tn:] for s_ in self.filterbank])
    FBm.filterbank = fbF
    CCm.FilterBank.transform_filterbank = tfb
    for i_ in range(5):
        fbF(np.zeros((C, st + Tn)), rfs, i_)                                  # thiet ke truoc khi do

    def doc(s):
        X, y, _, _ = nap(s)
        d = X[:, :, : st + Tn] - X[:, :, :ON].mean(axis=2, keepdims=True)
        L = d.reshape(6, NT, C, -1).transpose(1, 0, 2, 3)                      # label*block*C*Tdai
        return L[..., st:], L
    trw, trl, trlong = [], [], []
    for s in (17, 27):                                                         # ngoai tuyen (chay_fold dong 146-190), 0 vong huan luyen
        w, L = doc(s)
        lst = np.zeros((w.shape[0], w.shape[1], 2 * Nh, w.shape[3]))
        for cla in range(NT):
            P_ = FN.GLST(w[cla], rfs, Nh, config.list_freqs[cla], config.list_phase[cla])
            for blo in range(w.shape[1]):
                lst[cla, blo:, :, :] = P_ @ w[cla, blo]
        trw.append(w); trl.append(lst); trlong.append(L)
    d1raw, d1lst, d1long = (np.concatenate(v, axis=1) for v in (trw, trl, trlong))
    nb = d1raw.shape[1]
    train_raw = np.zeros((NT * nb, C, Tn)); train_lst = np.zeros((NT * nb, 2 * Nh, Tn)); train_label = np.zeros(NT * nb); train_long = np.zeros((NT * nb, C, d1long.shape[3]))
    for j in range(NT):
        sl = slice(nb * j, nb * j + nb)
        train_raw[sl] = d1raw[j]; train_lst[sl] = d1lst[j]; train_label[sl] = j; train_long[sl] = d1long[j]
    _, Lst_weights, data_mean = FN.get_LST_transform(train_raw, train_lst, train_label, list_freqs=config.list_freqs, fs=config.rfs, num_harms=4)
    torch.manual_seed(seed)
    um = Mm.universal_model(); model = um.siamese
    wp = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]; ws = [(3, 92), (12, 92), (20, 92), (28, 92), (36, 92)]
    filterbank = FN.generate_filterbank(wp, ws, srate=rfs, order=15, rp=0.5)
    fw = np.array([(i_ + 1) ** (-1.25) + 0.25 for i_ in range(5)])
    e2 = CCm.FBTtCCA(filterbank=filterbank, n_components=1, filterweights=fw, n_jobs=-1).fit(train_long, train_label, Yf=data_mean)
    yref1 = FBm.cca_reference(config.list_freqs, config.rfs, config.T, config.Nh)
    mm = lambda v: (v - v.min(1, keepdims=True)) / (v.max(1, keepdims=True) - v.min(1, keepdims=True))

    def chay(wb, Lb):                                                          # 'final' cua cca_net_tai_hien (dong 175-208), bo e1/f4
        n = wb.shape[0]
        dt = fb_nhan_qua(Lb[:, None])[:, 0]
        tl = np.zeros((n, NT, 2 * Nh, Tn))
        for cla in range(NT):
            P_ = Lst_weights[cla]
            for i in range(n):
                tl[i, cla] = P_ @ wb[i]
        dt = FN.FB_stand(dt); tl = FN.FB_stand(tl)
        pr = np.asarray(um.test(model, dt, tl)).reshape(n, NT)
        _, f2 = e2.predict(Lb)
        _, f1 = FBm.fbcca(Lb, y_ref=yref1); _, f3 = FBm.fbcca(Lb, y_ref=data_mean)
        f1, f2, f3 = np.asarray(f1, float), np.asarray(f2, float), np.asarray(f3, float)
        hong = ~(np.isfinite(pr).all(1) & np.isfinite(f1).all(1) & np.isfinite(f2).all(1) & np.isfinite(f3).all(1)
                 & (np.ptp(pr, 1) > 0) & (np.ptp(f1, 1) > 0) & (np.ptp(f2, 1) > 0) & (np.ptp(f3, 1) > 0))
        HONG["n"] += int(hong.sum())
        f1, f2, f3, pr = mm(f1), mm(f2), mm(f3), mm(pr)
        return np.argmax(pr + (f1 + f2 + f3) / 3, -1)

    def quyet(xr):
        d = xr[:, : st + Tn] - xr[:, :ON].mean(axis=1, keepdims=True)
        return int(chay(d[None, :, st:], d[None])[0])

    # ---- doi chung DOC LAP: ba ham loc trich NGUYEN VAN tu cca_net_tai_hien.py bang AST (thiet ke bo loc moi loi goi)
    #      khoi thu dong 175-208 cua ma do (ke ca e1/f4), tinh tren khoi 1 (40 luot = 40 lop) cua nguoi 1 ----
    import ast
    src_th = open(os.path.join(G, "cca_net_tai_hien.py"), encoding="utf-8").read()
    ns_g = dict(np=np, signal=signal, config=config, rfs=rfs, T=Tn)
    for nd in ast.parse(src_th).body:
        if isinstance(nd, ast.FunctionDef) and nd.name in ("fb_nhan_qua", "fbcca_filterbank_nhan_qua", "tfb_nhan_qua"):
            exec(compile(ast.Module(body=[nd], type_ignores=[]), "cca_net_tai_hien.py", "exec"), ns_g)
    Yf = FN.generate_cca_references(config.list_freqs, srate=rfs, T=Tn / rfs, phases=config.list_phase, n_harmonics=5)
    e1 = CCm.FBTtCCA(filterbank=filterbank, n_components=1, filterweights=fw, n_jobs=-1).fit(train_long, train_label, Yf=Yf)

    def goc(Xr):
        assert len(Xr) == NT, "doi chung goc can dung 40 luot cua khoi 1"
        cu = (FBm.filterbank, CCm.FilterBank.transform_filterbank)
        FBm.filterbank = ns_g["fbcca_filterbank_nhan_qua"]; CCm.FilterBank.transform_filterbank = ns_g["tfb_nhan_qua"]
        try:
            d = Xr[:, :, : st + Tn] - Xr[:, :, :ON].mean(axis=2, keepdims=True)
            L = d[:, None]; w = L[..., st:]                                    # label*block (block = 1)
            data1c_test = ns_g["fb_nhan_qua"](L)
            test_new = np.zeros((w.shape[0], w.shape[1], NT, 2 * Nh, w.shape[3]))
            for cla in range(NT):
                P_ = Lst_weights[cla]
                for idx in range(w.shape[0]):
                    for blo in range(w.shape[1]):
                        test_new[idx, blo, cla] = P_ @ w[idx, blo]
            data_test = data1c_test[:, 0]; data_test_lst = test_new[:, 0]; data_test_long = L[:, 0]
            data_test = FN.FB_stand(data_test); data_test_lst = FN.FB_stand(data_test_lst)
            prediction = um.test(model, data_test, data_test_lst)
            _, f4 = e1.predict(data_test_long); p2, f2 = e2.predict(data_test_long)
            _, f1 = FBm.fbcca(data_test_long, y_ref=FBm.cca_reference(config.list_freqs, config.rfs, config.T, config.Nh))
            p3, f3 = FBm.fbcca(data_test_long, y_ref=data_mean)
            f1, f2, f3, f4, pr = mm(f1), mm(f2), mm(f3), mm(f4), mm(prediction)
            return np.argmax(pr + (f1 + f2 + f3) / 3, -1)
        finally:
            FBm.filterbank, CCm.FilterBank.transform_filterbank = cu
    def lo(Xr):                                                                # chia khoi 40 luot
        assert len(Xr) % NT == 0, "so luot kiem CCA-Net phai la boi cua 40"
        return np.concatenate([goc(Xr[k: k + NT]) for k in range(0, len(Xr), NT)])
    return quyet, lo, None


def moi_truong(pp):
    v = dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0])
    for mn in ("scipy", "sklearn", "joblib", "threadpoolctl"):
        mo = sys.modules.get(mn)
        if mo is None:
            try:
                mo = importlib.import_module(mn)
            except Exception:
                mo = None
        v[mn] = getattr(mo, "__version__", None) if mo else None
    tep = ["b1_diem.py", "global_dnn.py", "zcu104_ktnr_daucuoi.py", "rcs_tai_hien.py", "ensemble_dnn.py", "global_dnn_thichnghi.py",
           "du_lieu_ben/benchmark_S1.npz", "du_lieu_ben/benchmark_S17.npz", "du_lieu_ben/benchmark_S27.npz", "trongso_T10/fold017_trongso.pt",
           "cca_net/shim_zcu104/config.py", "cca_net/cca_net_tai_hien.py"]
    v["sha256"] = {f: sha(os.path.join(HERE, f))[:16] for f in tep if os.path.exists(os.path.join(HERE, f))}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", required=True, choices=["ktnr", "fbcca", "rcs", "ed22", "edgoc", "ccanet"])
    ap.add_argument("--T", type=float, required=True); ap.add_argument("--luong", type=int, required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--khoi-dong", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261107); ap.add_argument("--kiem", action="store_true")
    ap.add_argument("--kiem-luot", type=int, default=240)
    ap.add_argument("--phien", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); np.random.seed(a.seed)
    ngu = torch.no_grad() if a.pp != "ccanet" else contextlib.nullcontext()
    t_lap = time.perf_counter()
    f = {"ktnr": lam_ktnr, "fbcca": lam_fbcca, "rcs": lam_rcs, "ccanet": lam_ccanet,
         "ed22": functools.partial(lam_ed, nch=22), "edgoc": functools.partial(lam_ed, nch=120)}[a.pp]
    quyet, lo, phu = f(a.T, a.seed)
    t_lap = time.perf_counter() - t_lap
    Xr = nap(1)[0]
    base = dict(phan="doithu_dau_cuoi", pp=a.pp, T=a.T, luong=a.luong, seed=a.seed, phien=a.phien,
                script_sha256=sha(os.path.abspath(__file__)), torch_luong=torch.get_num_threads(), moi_truong=moi_truong(a.pp),
                bien_moi_truong={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
                cuda=torch.cuda.is_available(), giay_dung_ngoai_tuyen=t_lap)
    ghi = (lambda s: open(a.out, "w").write(s + "\n")) if a.out else print
    if a.kiem:
        nk = a.kiem_luot
        with ngu:
            d1 = np.asarray([quyet(Xr[i]) for i in range(nk)])
            d2 = np.asarray(lo(Xr[:nk]))
        khac = int((d1 != d2).sum())
        ok = khac == 0 and HONG["n"] == 0 and len(d1) == nk and ((d1 >= 0) & (d1 < N)).all()
        ghi(json.dumps(dict(base, kiem="tung luot so voi theo lo", n=nk, so_khac=khac, chi_so_khac=np.where(d1 != d2)[0][:20].tolist(),
                            luot_diem_hong=HONG["n"], dat=bool(ok))))
        sys.exit(0 if ok else 5)
    tt = []
    with ngu:
        for i in range(a.khoi_dong + a.n):
            xr = Xr[i % len(Xr)]
            t0 = time.perf_counter(); quyet(xr); t1 = time.perf_counter()
            if i >= a.khoi_dong:
                tt.append(t1 - t0)
        out = dict(base, n=a.n, khoi_dong=a.khoi_dong, tong=q(tt), luot_diem_hong=HONG["n"])
        if a.pp == "rcs":                                                      # chi phi hoc lai moi lo 20 luot, trang thai tich luy (bao rieng)
            out["hoc_lai_lo20_tich_luy"] = q(phu(Xr))
    ghi(json.dumps(out))


if __name__ == "__main__":
    main()
