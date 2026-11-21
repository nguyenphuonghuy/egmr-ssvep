#!/usr/bin/env python
"""Do E4R-b (E4 tren nen KTN-R) tren PS cua ZCU104.
In MOT dong JSON. Can cung thu muc: global_dnn.py, b1_diem.py, zcu104_b1gop_eg.py; du lieu du_lieu_ben/benchmark_S1.npz.
Trong so ngau nhien (chi do thoi gian, bo nho), nhu cong R cua v4 va phan b1 cua thi nghiem B. Lap lai 240 luot tho cua nguoi 1.
Moi luot (dong ho bat dau TRUOC GD.tien_xu_ly, gom thiet ke bo loc trong loi goi):
  quyet dinh: bank loc -> luot xuoi giu do thi (W_l2, b_l2 la la co grad) -> nhanh R (NumPy float64 nhu ktnr_buoc1: y = w_R^T u, tru TB,
              R = |Q_c^T y|/|y|, z) -> s = z(logit) + ALPHA z(R) -> argmax                                                     [t_quyet_dinh]
  nen       : mat mat -do_tin(z(logit) + ALPHA z(R)) (z(R) hang so), grad theo (W_l2, b_l2), g = D^T grad, EG, lam = softmax, gop W va b   [t_nen]
Mot lan moi phien (do rieng, --n lan): dung co so sin/cos Qb (40 lop, H = 5, QR) va w_R (vecto rieng lon nhat cua TB 5 bo loc nguon 9 x 9).
Bo nho them phan thich nghi: storage duy nhat cua tensor autograd giu lai (saved_tensors_hooks, loai tham so mang nen) + tensor tam + trang thai
(W0, b0, K so gia, theta, w_R, z(R) cua luot). Qb bao rieng (tham chieu tinh cua KTN-R). VmHWM bao kem.
"""
import argparse, io, json, math, os, sys, time, contextlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FS, ALPHA, H = 250, 0.5, 5


def q(v):
    v = np.asarray(v, float)
    return dict(trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)), max_ms=1e3 * float(v.max()),
                tb_ms=1e3 * float(v.mean()), n=int(len(v)), vuot_50ms=float((v > 0.05).mean()), vuot_500ms=float((v > 0.5).mean()))


def co_so(freqs, fs, ns):                                                # ban sao ktnr_buoc1.co_so
    t = np.arange(ns) / fs
    out = []
    for f in freqs:
        Y = np.stack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)], 1)
        out.append(np.linalg.qr(Y)[0])
    return np.stack(out)


def w_tu(ws):                                                            # ban sao ktnr_buoc1.w_tu (dau_max: vecto dau tien cua tri lon nhat)
    M = sum(np.outer(w, w) for w in ws) / len(ws)
    lam, V = np.linalg.eigh(M)
    m = lam.max()
    return V[:, int(np.where(lam >= m - 1e-12 * max(1.0, abs(m)))[0][0])]


def diem_R1(yv, Qb):
    """Mot luot: yv (ns,) da tru TB -> (N,)."""
    p = np.einsum("t,ntk->nk", yv, Qb)
    ny = np.linalg.norm(yv)
    return np.sqrt((p ** 2).sum(-1)) / ny if ny > 0 else np.zeros(Qb.shape[0])


def z_np(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def z_t(v):
    return (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True, unbiased=False) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--luong", type=int, default=1)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--khoi-dong", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261107); ap.add_argument("--tien-trinh", type=int, default=0)
    ap.add_argument("--du-lieu", default="du_lieu_ben")
    a = ap.parse_args()
    import global_dnn as GD
    import zcu104_b1gop_eg as Z
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    zf = np.load(os.path.join(a.du_lieu, "benchmark_S1.npz")); Xr = zf["X"]; freqs = zf["freqs"]
    ns, C, K, N = int(round(a.T * FS)), 9, 5, 40
    m = Z.GlobalDNN(C, ns, N).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    dem = Z.DemLuu([p.untyped_storage().data_ptr() for p in m.parameters()])
    W0, b0 = m.l2.weight.clone(), m.l2.bias.clone()
    DW = 0.01 * torch.randn(K, W0.numel()); DB = 0.01 * torch.randn(K, b0.numel())
    theta = torch.zeros(K); Gm = 0.0
    W, b = W0 + DW.mean(0).view_as(W0), b0 + DB.mean(0)
    ws = [v / np.linalg.norm(v) for v in rng.standard_normal((K, C))]    # 5 bo loc nguon (ngau nhien; TRCA nguon lam ngoai tuyen)
    tp = []
    for i in range(a.khoi_dong + a.n):                                   # mot lan moi phien, do rieng
        t0 = time.perf_counter(); Qb = co_so(freqs, FS, ns); wR = w_tu(ws); t1 = time.perf_counter()
        if i >= a.khoi_dong:
            tp.append(t1 - t0)
    tb, tR, tq, tn, tt = [], [], [], [], []
    kich_hoat, tam = 0, 0
    for i in range(a.khoi_dong + a.n):
        xr = Xr[i % len(Xr)][None]
        t0 = time.perf_counter()
        x = torch.tensor(GD.tien_xu_ly(xr, "benchmark", "causal", ns, FS))[0][None]
        t1 = time.perf_counter()
        dem.reset()
        Wl, bl = W.detach().requires_grad_(True), b.detach().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(dem.goi, lambda t: t):
            with torch.no_grad():
                h1 = m.l1(x)
            lg = m.sau(h1, Wl, bl)
            tr0 = time.perf_counter()
            U = h1[0, 0].double().numpy()
            yv = wR @ U; yv = yv - yv.mean()
            zR = z_np(diem_R1(yv, Qb))
            s = z_np(lg.detach().double().numpy()[0]) + ALPHA * zR
            int(s.argmax())
            t2 = time.perf_counter()
            zRt = torch.tensor(zR, dtype=torch.float32)[None]
            mm = -Z.do_tin(z_t(lg) + ALPHA * zRt).sum()
        gW, gb = torch.autograd.grad(mm, [Wl, bl])
        g = DW @ gW.flatten() + DB @ gb
        Gm = max(Gm, float(g.abs().max()))
        if Gm > 0:
            theta = theta - math.sqrt(2.0 * math.log(K) / (Gm * Gm * 200)) * g
        lam = torch.softmax(theta, 0)
        W = W0 + (lam @ DW).view_as(W0); b = b0 + lam @ DB
        t3 = time.perf_counter()
        if i >= a.khoi_dong:
            tb.append(t1 - t0); tR.append(t2 - tr0); tq.append(t2 - t0); tn.append(t3 - t2); tt.append(t3 - t0)
            kich_hoat = max(kich_hoat, dem.duy_nhat); tam = max(tam, Z.nb(gW, gb, g, lam, W, b, zRt))
    nb_np = lambda *v: sum(x.nbytes for x in v)
    trang_thai = Z.nb(W0, b0, DW, DB, theta) + nb_np(wR)
    n2 = ns // 2; nch = 22
    mac_mang = 3 * C * ns + nch * C * ns + nch * n2 * nch * 2 + nch * n2 * nch * 10 + nch * n2 * N
    mac_nguoc = nch * n2 * N + nch * n2 * nch * 10 + nch * n2 * nch * 2 + nch * C * ns
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    print(json.dumps(dict(phan="b2e4r", T=a.T, luong=a.luong, tien_trinh=a.tien_trinh, seed=a.seed, n=a.n, khoi_dong=a.khoi_dong,
                          torch=torch.__version__, numpy=np.__version__, torch_luong=torch.get_num_threads(),
                          env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                          ket_qua=dict(bank_loc=q(tb), nhanh_R=q(tR), quyet_dinh_tu_tho=q(tq), nen=q(tn), tong_cung_luot=q(tt), mot_lan_moi_phien=q(tp)),
                          mac=dict(suy_luan_mang=mac_mang, nhanh_R=C * ns + N * 2 * H * ns + ns + 2 * H * N, z_gop=6 * N,
                                   nen=mac_nguoc + 2 * K * (nch * C + nch), ghi_chu="du toan theo cong thuc lop; khong gom bank loc"),
                          byte=dict(kich_hoat_duy_nhat=kich_hoat, tam=tam, trang_thai=trang_thai, them_thich_nghi=kich_hoat + tam + trang_thai,
                                    co_so_Qb_float64=nb_np(Qb)),
                          vmhwm_kb=Z.vmhwm_kb(), np_config=buf.getvalue()[-800:])))


if __name__ == "__main__":
    main()
