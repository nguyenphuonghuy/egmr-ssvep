#!/usr/bin/env python
"""Do tre KTN-R tren ZCU104 (PS, Cortex-A53) cho bai B1..
Mo rong zcu104_ktn_dotre.py (cung mang nen, cung pham vi: khong gom bank loc va tien xu ly tin hieu tho; batch 1, eval, no_grad).
  luot khoi tao 1..40: nhu KTN (quyet dinh bang mang nen + cham S mo-dun + cong don do tin).
  luot 40 them: TB tham so 5 mo-dun do tin cao nhat (nhu KTN) + w_R = vecto rieng lon nhat cua TB w_s w_s^T (5 bo loc nguon 9 x 9).
  luot suy luan: mang da gop + nhanh R (NumPy float64 nhu ktnr_buoc1: y = w_R^T u, tru TB, R_c = |Q_c^T y|/|y|) + z(logit) + 0,5 z(R) + argmax.
Co so sin/cos Q_c chi phu thuoc tan so, fs, so mau: dung ngoai tuyen, do rieng (--n-qb lan). Trong so va bo loc nguon ngau nhien.
Ghi JSON mot dong ra stdout. Tu chua, chi can torch + numpy.
"""
import argparse, json, time
import numpy as np
import torch
from zcu104_ktn_dotre import GlobalDNN, do_tin

FS, ALPHA, H = 250, 0.5, 5


def co_so(freqs, fs, ns):                                                # ban sao ktnr_buoc1.co_so
    t = np.arange(ns) / fs
    out = []
    for f in freqs:
        Y = np.stack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)], 1)
        out.append(np.linalg.qr(Y)[0])
    return np.stack(out)


def w_tu(ws):                                                            # ban sao ktnr_buoc1.w_tu (dau_max)
    M = sum(np.outer(w, w) for w in ws) / len(ws)
    lam, V = np.linalg.eigh(M)
    m = lam.max()
    return V[:, int(np.where(lam >= m - 1e-12 * max(1.0, abs(m)))[0][0])]


def diem_R1(yv, Qb):
    p = np.einsum("t,ntk->nk", yv, Qb)
    ny = np.linalg.norm(yv)
    return np.sqrt((p ** 2).sum(-1)) / ny if ny > 0 else np.zeros(Qb.shape[0])


def z_np(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--luong", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--n-sau", type=int, default=100)
    ap.add_argument("--n-qb", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    C, ns = 9, int(round(a.T * FS))
    freqs = np.linspace(8.0, 15.8, a.N)
    m = GlobalDNN(C, ns, a.N).eval()
    dW = [0.01 * torch.randn_like(m.l2.weight) for _ in range(a.S)]; db = [0.01 * torch.randn_like(m.l2.bias) for _ in range(a.S)]
    Ws = [m.l2.weight + d for d in dW]; bs = [m.l2.bias + d for d in db]
    wsrc = [v / np.linalg.norm(v) for v in rng.standard_normal((a.S, C))]   # bo loc TRCA nguon (lam ngoai tuyen; o day ngau nhien)
    X = torch.randn(40 + a.n_sau, 1, 3, C, ns)
    t_qb = []
    for _ in range(a.n_qb):                                              # ngoai tuyen, do rieng
        t0 = time.perf_counter(); Qb = co_so(freqs, FS, ns); t_qb.append(time.perf_counter() - t0)
    t_kd, t_sau, t_R = [], [], []
    acc = torch.zeros(a.S)
    with torch.no_grad():
        for _ in range(5):                                                 # khoi dong nong (khong tinh)
            m(X[0]); m.sau(m.l1(X[0]), Ws[0], bs[0])
            w_tu(wsrc[:5]); diem_R1(np.ones(ns), Qb)
        for i in range(40):
            t0 = time.perf_counter()
            x = X[i]
            d = int(m(x).argmax(1))
            h1 = m.l1(x)
            for s in range(a.S):
                acc[s] += do_tin(m.sau(h1, Ws[s], bs[s]))[0]
            if i == 39:
                top = torch.argsort(-acc, stable=True)[:5]
                Wc = m.l2.weight + torch.stack([dW[int(j)] for j in top]).mean(0)
                bc = m.l2.bias + torch.stack([db[int(j)] for j in top]).mean(0)
                t5 = time.perf_counter()
                wR = w_tu([wsrc[int(j)] for j in top])
                t_wR = time.perf_counter() - t5
            t_kd.append(time.perf_counter() - t0)
        for i in range(40, 40 + a.n_sau):
            t0 = time.perf_counter()
            h1 = m.l1(X[i])
            lg = m.sau(h1, Wc, bc)
            tr0 = time.perf_counter()
            U = h1[0, 0].double().numpy()
            yv = wR @ U; yv = yv - yv.mean()
            s = z_np(lg.double().numpy()[0]) + ALPHA * z_np(diem_R1(yv, Qb))
            d = int(s.argmax())
            t1 = time.perf_counter()
            t_sau.append(t1 - t0); t_R.append(t1 - tr0)
    n2 = ns // 2
    mac_mang = 3 * C * ns + 22 * C * ns + 22 * n2 * 22 * 2 + 22 * n2 * 22 * 10 + 22 * n2 * a.N
    mac_R = C * ns + a.N * 2 * H * ns + ns + 2 * H * a.N
    q = lambda v: dict(tb_ms=1e3 * float(np.mean(v)), trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)),
                       max_ms=1e3 * float(np.max(v)), n=len(v))
    print(json.dumps(dict(phan="ktnr", S=a.S, N=a.N, T=a.T, luong=a.luong, seed=a.seed, torch=torch.__version__, numpy=np.__version__,
                          torch_luong=torch.get_num_threads(), khoi_tao=q(t_kd), luot_40_ms=1e3 * t_kd[-1], w_R_ms=1e3 * t_wR,
                          suy_luan=q(t_sau), nhanh_R_va_gop=q(t_R), dung_Qb_ngoai_tuyen=q(t_qb),
                          mac_suy_luan=mac_mang + mac_R + 6 * a.N, mac_mang=mac_mang, mac_nhanh_R=mac_R, byte_Qb_float64=int(Qb.nbytes))))


if __name__ == "__main__":
    main()
