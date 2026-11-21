#!/usr/bin/env python
"""Do thoi gian DAU-CUOI cua KTN-R tren PS cua ZCU104.
Chay trong ~/ssvep_zcu104_pkg/doi_thu (can global_dnn.py, b1_diem.py,
du_lieu_ben/benchmark_S1.npz, trongso_T10/fold017_trongso.pt). In MOT dong JSON.
Moi luot (dong ho bat dau khi luot tho 9 x 1500 da o bo nho): tru TB truoc kich thich -> bank 3 dai sosfilt (he so thiet ke MOT lan luc khoi dong)
-> cat cua so -> mang da gop (no_grad; trong so nen that + so gia ngau nhien nho) -> nhanh R (NumPy float64 nhu ktnr_buoc1) -> z gop -> argmax.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from scipy.signal import sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FS, ALPHA, H = 250, 0.5, 5


def co_so(freqs, fs, ns):                                                # ban sao ktnr_buoc1.co_so
    t = np.arange(ns) / fs
    return np.stack([np.linalg.qr(np.stack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)], 1))[0] for f in freqs])


def diem_R1(yv, Qb):
    p = np.einsum("t,ntk->nk", yv, Qb)
    ny = np.linalg.norm(yv)
    return np.sqrt((p ** 2).sum(-1)) / ny if ny > 0 else np.zeros(Qb.shape[0])


def z_np(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def q(v):
    v = np.asarray(v, float)
    return dict(tb_ms=1e3 * float(v.mean()), trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)),
                max_ms=1e3 * float(v.max()), n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, required=True); ap.add_argument("--luong", type=int, required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--khoi-dong", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    import global_dnn as GD
    import b1_diem as B
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    z = np.load("du_lieu_ben/benchmark_S1.npz"); Xr = z["X"]; freqs = z["freqs"]
    c = B.DS["benchmark"]; on = c["onset"]; st = on + int(round(c["lat"] * FS)); ns = int(round(a.T * FS)); C, N = 9, 40
    _, sos = GD.bo_loc(FS)                                                 # thiet ke MOT lan
    m = GD.GlobalDNN(C, ns, N, nch=22)
    sd = torch.load("trongso_T10/fold017_trongso.pt", map_location="cpu")
    nap_that = False
    if a.T == 1.0:
        m.load_state_dict(sd); nap_that = True                             # trong so that chi co cho cua so 1,0 s
    m.eval()
    with torch.no_grad():
        m.l2.weight += 0.01 * torch.randn_like(m.l2.weight); m.l2.bias += 0.01 * torch.randn_like(m.l2.bias)   # mo-dun da gop
    w = rng.standard_normal(C); wR = w / np.linalg.norm(w)
    Qb = co_so(freqs, FS, ns)
    tb, tl, tt = [], [], []
    with torch.no_grad():
        for i in range(a.khoi_dong + a.n):
            xr = Xr[i % len(Xr)]
            t0 = time.perf_counter()
            xb = xr[:, : st + ns] - xr[:, :on].mean(axis=1, keepdims=True)
            u = np.empty((len(sos), C, ns), np.float32)
            for r, so in enumerate(sos):
                u[r] = sosfilt(so, xb, axis=1)[:, st: st + ns]
            t1 = time.perf_counter()
            h1 = m.l1(torch.from_numpy(u)[None])
            lg = m.fc(m.l4(torch.relu(m.l3(m.l2(h1)))).flatten(1))          # = GD.GlobalDNN.forward o che do eval, dung lai h1
            U = h1[0, 0].double().numpy()
            yv = wR @ U; yv = yv - yv.mean()
            s = z_np(lg.double().numpy()[0]) + ALPHA * z_np(diem_R1(yv, Qb))
            d = int(s.argmax())
            t2 = time.perf_counter()
            if i >= a.khoi_dong:
                tb.append(t1 - t0); tl.append(t2 - t1); tt.append(t2 - t0)
    print(json.dumps(dict(phan="ktnr_dau_cuoi", T=a.T, luong=a.luong, seed=a.seed, n=a.n, trong_so_nen_that=nap_that,
                          torch=torch.__version__, numpy=np.__version__, torch_luong=torch.get_num_threads(),
                          bank_loc=q(tb), loi=q(tl), tong=q(tt))))


if __name__ == "__main__":
    main()
