#!/usr/bin/env python
"""Do pha khoi dong cua EGM-R tren PS cua ZCU104 (bo sung 2026-11-20). In MOT dong JSON.
Can cung thu muc: global_dnn.py, zcu104_b1gop_eg.py; du lieu du_lieu_ben/benchmark_S1.npz (nhu zcu104_b2e4r.py).
Trong so ngau nhien (chi do thoi gian, bo nho). Lap lai 240 luot tho cua nguoi 1.
Moi luot khoi dong (dong ho bat dau TRUOC GD.tien_xu_ly, gom thiet ke bo loc trong loi goi):
  bank loc -> l1 (dung chung) -> quyet dinh bang mang chung (W0, b0) -> voi MOI nguon s = 1..S: l2 voi W0 + dW_s, b0 + db_s, l3, l4, FC,
  do tin c(logit_s) cong don (khong giu do thi, torch.no_grad)                                                         [t_luot]
Cuoi pha khoi dong (do rieng, --n lan): TB do tin, chon 5 nguon, xep D (5 so gia), dung co so sin/cos Qb va w_R tu 5 bo loc nguon [t_chuyen].
Bo nho luu cho pha khoi dong: S so gia l2 (float32), S bo loc TRCA nguon (float64, 9 so), tham so mang nen; VmHWM bao kem.
"""
import argparse, io, json, os, sys, time, contextlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FS, H, K, N0 = 250, 5, 5, 40


def q(v):
    v = np.asarray(v, float)
    return dict(trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)), max_ms=1e3 * float(v.max()),
                tb_ms=1e3 * float(v.mean()), n=int(len(v)), vuot_500ms=float((v > 0.5).mean()))


def co_so(freqs, fs, ns):                                                # ban sao zcu104_b2e4r.co_so
    t = np.arange(ns) / fs
    out = []
    for f in freqs:
        Y = np.stack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)], 1)
        out.append(np.linalg.qr(Y)[0])
    return np.stack(out)


def w_tu(ws):                                                            # ban sao zcu104_b2e4r.w_tu
    M = sum(np.outer(w, w) for w in ws) / len(ws)
    lam, V = np.linalg.eigh(M)
    m = lam.max()
    return V[:, int(np.where(lam >= m - 1e-12 * max(1.0, abs(m)))[0][0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--luong", type=int, default=1); ap.add_argument("--n", type=int, default=200); ap.add_argument("--khoi-dong", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261120); ap.add_argument("--tien-trinh", type=int, default=0)
    ap.add_argument("--du-lieu", default="du_lieu_ben")
    a = ap.parse_args()
    import global_dnn as GD
    import zcu104_b1gop_eg as Z
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    zf = np.load(os.path.join(a.du_lieu, "benchmark_S1.npz")); Xr = zf["X"]; freqs = zf["freqs"]
    ns, C, N = int(round(a.T * FS)), 9, 40
    m = Z.GlobalDNN(C, ns, N).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W0, b0 = m.l2.weight.clone(), m.l2.bias.clone()
    DW = 0.01 * torch.randn(a.S, W0.numel()); DB = 0.01 * torch.randn(a.S, b0.numel())
    WS = [(W0 + DW[s].view_as(W0), b0 + DB[s]) for s in range(a.S)]    # mo-dun nguon dung san (W0 + so gia), nhu khi trien khai
    ws_nguon = [v / np.linalg.norm(v) for v in rng.standard_normal((a.S, C))]
    tl, tb_, tc = [], [], []
    g = torch.zeros(a.S)
    for i in range(a.khoi_dong + a.n):
        xr = Xr[i % len(Xr)][None]
        t0 = time.perf_counter()
        x = torch.tensor(GD.tien_xu_ly(xr, "benchmark", "causal", ns, FS))[0][None]
        t1 = time.perf_counter()
        with torch.no_grad():
            h1 = m.l1(x)
            int(m.sau(h1, W0, b0).argmax())                              # quyet dinh pha khoi dong bang mang chung
            for s, (Ws, bs) in enumerate(WS):
                g[s] += float(Z.do_tin(m.sau(h1, Ws, bs))[0])
        t2 = time.perf_counter()
        if i >= a.khoi_dong:
            tb_.append(t1 - t0); tl.append(t2 - t0)
    for i in range(a.khoi_dong + a.n):                                   # chuyen pha o luot 40 (mot lan moi phien)
        t0 = time.perf_counter()
        top = torch.argsort(-(g / N0), stable=True)[:K]
        D = torch.stack([DW[int(j)] for j in top]); Db = torch.stack([DB[int(j)] for j in top])
        Qb = co_so(freqs, FS, ns); wR = w_tu([ws_nguon[int(j)] for j in top])
        t1 = time.perf_counter()
        if i >= a.khoi_dong:
            tc.append(t1 - t0)
    nb_t = lambda *v: sum(x.numel() * x.element_size() for x in v)
    luu = dict(so_gia_nguon=nb_t(DW, DB), bo_loc_nguon_float64=int(sum(w.nbytes for w in ws_nguon)),
               mang_nen=int(sum(p.numel() * p.element_size() for p in m.parameters())), so_tham_so_mang=int(sum(p.numel() for p in m.parameters())))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    print(json.dumps(dict(phan="khoi_dong", T=a.T, S=a.S, luong=a.luong, tien_trinh=a.tien_trinh, seed=a.seed, n=a.n, khoi_dong=a.khoi_dong,
                          torch=torch.__version__, numpy=np.__version__, torch_luong=torch.get_num_threads(),
                          ket_qua=dict(bank_loc=q(tb_), luot_khoi_dong=q(tl), chuyen_pha_luot_40=q(tc)), byte=luu,
                          vmhwm_kb=Z.vmhwm_kb(), np_config=buf.getvalue()[-400:])))


if __name__ == "__main__":
    main()
