#!/usr/bin/env python
"""Do tren PS cua ZCU104 (bo sung 2026-11-21, K). In MOT dong JSON. Can cung thu muc: global_dnn.py, zcu104_b1gop_eg.py,
zcu104_b2e4r.py (dung lai co_so, w_tu, diem_R1, z_np, z_t); du lieu du_lieu_ben/benchmark_S1.npz. Trong so ngau nhien (chi do thoi gian, bo nho).
--che-do mot_lan: chi phi MOT lan cua EGM-R_40 o cuoi pha khoi dong. Dau ra l1 va z(R) cua 40 luot khoi dong da giu san; chay T buoc gradient mu
         (luot thu k lay tu luot k mod 40: luot xuoi tu l2, mat mat -do_tin(z(logit) + ALPHA z(R)), grad theo (W_l2, b_l2), g = D^T grad, EG),
         roi gop W, b. Do --n lan. Bo nho: dau ra l1 cua 40 luot + z(R) 40 luot.
--che-do l2: cap nhat tung luot 220 tham so l2: nhu viec nen cua zcu104_b2e4r.py nhung buoc gradient xuong truc tiep tren (W_l2, b_l2)
         (khong co D^T grad, khong gop lai). Lap lai 240 luot tho cua nguoi 1, do --n luot sau --khoi-dong luot.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zcu104_b2e4r as Y        # noqa: E402
FS, ALPHA, H, N0 = 250, 0.5, 5, 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--che-do", required=True, choices=["mot_lan", "l2"])
    ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--Tphien", type=int, default=200)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--khoi-dong", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261121); ap.add_argument("--tien-trinh", type=int, default=0)
    ap.add_argument("--du-lieu", default="du_lieu_ben")
    a = ap.parse_args()
    import global_dnn as GD
    import zcu104_b1gop_eg as Z
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    zf = np.load(os.path.join(a.du_lieu, "benchmark_S1.npz")); Xr = zf["X"]; freqs = zf["freqs"]
    ns, C, K, N = int(round(a.T * FS)), 9, 5, 40
    m = Z.GlobalDNN(C, ns, N).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    dem = Z.DemLuu([p.untyped_storage().data_ptr() for p in m.parameters()])
    W0, b0 = m.l2.weight.clone(), m.l2.bias.clone()
    DW = 0.01 * torch.randn(K, W0.numel()); DB = 0.01 * torch.randn(K, b0.numel())
    ws = [v / np.linalg.norm(v) for v in rng.standard_normal((K, C))]
    Qb = Y.co_so(freqs, FS, ns); wR = Y.w_tu(ws)
    out = dict(phan="mot_lan_l2", che_do=a.che_do, T=a.T, Tphien=a.Tphien, tien_trinh=a.tien_trinh, seed=a.seed, n=a.n, khoi_dong=a.khoi_dong,
               torch=torch.__version__, numpy=np.__version__, torch_luong=torch.get_num_threads())
    if a.che_do == "mot_lan":
        h1s, zRs = [], []
        with torch.no_grad():
            for i in range(N0):
                x = torch.tensor(GD.tien_xu_ly(Xr[i][None], "benchmark", "causal", ns, FS))[0][None]
                h1 = m.l1(x); h1s.append(h1)
                U = h1[0, 0].double().numpy(); yv = wR @ U; yv = yv - yv.mean()
                zRs.append(torch.tensor(Y.z_np(Y.diem_R1(yv, Qb)), dtype=torch.float32)[None])
        tt = []
        for r in range(a.khoi_dong + a.n):
            t0 = time.perf_counter()
            theta = torch.zeros(K); Gm = 0.0
            for k in range(a.Tphien):
                lam = torch.softmax(theta, 0)
                Wl = (W0 + (lam @ DW).view_as(W0)).detach().requires_grad_(True); bl = (b0 + lam @ DB).detach().requires_grad_(True)
                lg = m.sau(h1s[k % N0], Wl, bl)
                mm = -Z.do_tin(Y.z_t(lg) + ALPHA * zRs[k % N0]).sum()
                gW, gb = torch.autograd.grad(mm, [Wl, bl])
                g = DW @ gW.flatten() + DB @ gb
                Gm = max(Gm, float(g.abs().max()))
                if Gm > 0:
                    theta = theta - math.sqrt(2.0 * math.log(K) / (Gm * Gm * a.Tphien)) * g
            lam = torch.softmax(theta, 0); W = W0 + (lam @ DW).view_as(W0); b = b0 + lam @ DB
            t1 = time.perf_counter()
            if r >= a.khoi_dong:
                tt.append(t1 - t0)
        out["ket_qua"] = dict(mot_lan=Y.q(tt))
        out["byte"] = dict(dau_ra_l1_40_luot=int(sum(h.numel() * h.element_size() for h in h1s)), zR_40_luot=int(sum(z.numel() * z.element_size() for z in zRs)))
    else:
        W, b = (W0 + DW.mean(0).view_as(W0)).clone(), (b0 + DB.mean(0)).clone(); Gm = 0.0
        Dkc = float(torch.cdist(torch.cat([DW, DB], 1), torch.cat([DW, DB], 1)).max())
        tn, tt, kich_hoat, tam = [], [], 0, 0
        for i in range(a.khoi_dong + a.n):
            xr = Xr[i % len(Xr)][None]
            t0 = time.perf_counter()
            x = torch.tensor(GD.tien_xu_ly(xr, "benchmark", "causal", ns, FS))[0][None]
            dem.reset()
            Wl, bl = W.detach().requires_grad_(True), b.detach().requires_grad_(True)
            with torch.autograd.graph.saved_tensors_hooks(dem.goi, lambda t: t):
                with torch.no_grad():
                    h1 = m.l1(x)
                lg = m.sau(h1, Wl, bl)
                U = h1[0, 0].double().numpy(); yv = wR @ U; yv = yv - yv.mean()
                zR = Y.z_np(Y.diem_R1(yv, Qb))
                s = Y.z_np(lg.detach().double().numpy()[0]) + ALPHA * zR
                int(s.argmax())
                t2 = time.perf_counter()
                zRt = torch.tensor(zR, dtype=torch.float32)[None]
                mm = -Z.do_tin(Y.z_t(lg) + ALPHA * zRt).sum()
            gW, gb = torch.autograd.grad(mm, [Wl, bl])
            Gm = max(Gm, float(torch.sqrt((gW ** 2).sum() + (gb ** 2).sum())))
            if Gm > 0:
                eta = Dkc / (Gm * math.sqrt(a.Tphien))
                W = W - eta * gW; b = b - eta * gb
            t3 = time.perf_counter()
            if i >= a.khoi_dong:
                tn.append(t3 - t2); tt.append(t3 - t0)
                kich_hoat = max(kich_hoat, dem.duy_nhat); tam = max(tam, Z.nb(gW, gb, W, b, zRt))
        out["ket_qua"] = dict(nen=Y.q(tn), tong_cung_luot=Y.q(tt))
        out["byte"] = dict(kich_hoat_duy_nhat=kich_hoat, tam=tam, trang_thai=Z.nb(W0, b0, W, b) + int(wR.nbytes),
                           them_thich_nghi=kich_hoat + tam + Z.nb(W0, b0, W, b) + int(wR.nbytes))
    out["vmhwm_kb"] = Z.vmhwm_kb()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
