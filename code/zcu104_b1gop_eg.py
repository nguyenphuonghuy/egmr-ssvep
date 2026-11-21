#!/usr/bin/env python
"""Cong R nhanh b1gop tren PS cua ZCU104 (Cortex-A53). Tu chua: chi can torch + numpy.
Ban 3 (them cong gradient/tam cua E6). Mang nen GlobalDNN thu nho (giong global_dnn.py), trong so ngau nhien (chi do thoi gian, bo nho).
Moi pha: 20 luot khoi dong may (khong tinh) + N_DO = 1.000 luot do (cau hinh: PyTorch, 1 luong la cau hinh quyet dinh).
  E4 sau luot 40: quyet dinh = forward voi W_l2 la la co grad (giu kich hoat) + argmax          [t_quyet_dinh]
                  nen = -do_tin; grad theo (W_l2, b_l2) qua FC, l4, l3, l2; g = D^T grad; EG; lam = softmax; gop W_l2 VA b_l2  [t_nen]
  E6 khoi dong:   quyet dinh = forward tai W0 co grad + argmax                                    [t_quyet_dinh]
                  nen = do_tin; grad tai W0; S tich vo huong voi kho S so gia; cong don              [t_nen]
                  luot 40 (do rieng, 1.000 lan): chon 5 nguon co diem cao nhat + gop W_l2 VA b_l2  [t_luot40]
Bo nho them (R3): storage DUY NHAT (data_ptr) cua cac tensor autograd giu trong forward + mat mat, LOAI storage cua tham so mang nen co san,
cuc dai qua cac luot; cong gradient va tensor tam (gW, gb, g, lam, W, b moi); trang thai (W0, b0, K so gia, theta) va kho S so gia bao rieng.
Bao kem: tong byte tho cua moi lan luu (co the trung storage), VmHWM tien trinh. MAC theo cong thuc lop (khong gom bank loc). In mot dong JSON.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class GlobalDNN(nn.Module):
    def __init__(self, C, ns, nclass, nch=22):
        super().__init__()
        self.l1 = nn.Conv2d(3, 1, (1, 1), bias=False)
        self.l2 = nn.Conv2d(1, nch, (C, 1))
        self.l3 = nn.Conv2d(nch, nch, (1, 2), stride=(1, 2))
        self.l4 = nn.Conv2d(nch, nch, (1, 10), padding="same")
        self.fc = nn.Linear(nch * (ns // 2), nclass)

    def sau(self, h1, W, b):
        h = Fn.conv2d(h1, W, b)
        h = self.l4(torch.relu(self.l3(h)))
        return self.fc(h.flatten(1))


def do_tin(lg):
    s, _ = torch.sort(lg, dim=-1); o = s[..., :-1]
    return (s[..., -1] - o.mean(-1)) / (o.std(-1, unbiased=False) + 1e-12)


def vmhwm_kb():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"):
            return int(l.split()[1])
    return -1


def q(v):
    v = np.asarray(v)
    return dict(tb_ms=1e3 * float(v.mean()), trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)),
                p99_ms=1e3 * float(np.percentile(v, 99)), max_ms=1e3 * float(v.max()), n=int(len(v)))


nb = lambda *ts: sum(t.numel() * t.element_size() for t in ts)


class DemLuu:
    """Dem storage duy nhat cua tensor autograd giu lai, loai storage cua tham so mang nen."""
    def __init__(self, loai):
        self.loai = set(loai); self.reset()

    def reset(self):
        self.ptr, self.duy_nhat, self.tho = set(), 0, 0

    def goi(self, t):
        self.tho += nb(t)
        st = t.untyped_storage(); p = st.data_ptr()
        if p not in self.loai and p not in self.ptr:
            self.ptr.add(p); self.duy_nhat += st.nbytes()
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--S", type=int, default=34)
    ap.add_argument("--luong", type=int, default=1); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--khoi-dong", type=int, default=20); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--T-eg", type=int, default=200, help="so luot sau n0 trong cong thuc eta (Benchmark 200)")
    a = ap.parse_args()
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed)
    C, ns, nch = 9, int(round(a.T * 250)), 22
    m = GlobalDNN(C, ns, a.N).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    dem = DemLuu([p.untyped_storage().data_ptr() for p in m.parameters()])
    W0, b0 = m.l2.weight.clone(), m.l2.bias.clone()
    DW = 0.01 * torch.randn(a.K, W0.numel()); DB = 0.01 * torch.randn(a.K, b0.numel())          # K so gia cua E4
    DWs = 0.01 * torch.randn(a.S, W0.numel()); DBs = 0.01 * torch.randn(a.S, b0.numel())        # kho S so gia (pha khoi dong E6)
    theta = torch.zeros(a.K); G = 0.0
    W, b = W0 + DW.mean(0).view_as(W0), b0 + DB.mean(0)
    X = torch.randn(a.khoi_dong + a.n, 1, 3, C, ns)

    # ---- E4, pha sau luot 40
    tq, tn, tr, kich_hoat, tho, tam = [], [], [], 0, 0, 0
    for i in range(a.khoi_dong + a.n):
        x = X[i]
        with torch.no_grad():                                            # tham chieu: suy luan thuan khong giu do thi
            t0 = time.perf_counter(); int(m.sau(m.l1(x), W, b).argmax(1)); r_ = time.perf_counter() - t0
        dem.reset()
        t0 = time.perf_counter()
        Wl, bl = W.detach().requires_grad_(True), b.detach().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(dem.goi, lambda t: t):
            with torch.no_grad():
                h1 = m.l1(x)
            lg = m.sau(h1, Wl, bl)
            int(lg.argmax(1))
            t1 = time.perf_counter()
            mm = -do_tin(lg).sum()                                       # mat mat nam trong pham vi dem
        gW, gb = torch.autograd.grad(mm, [Wl, bl])
        g = DW @ gW.flatten() + DB @ gb
        G = max(G, float(g.abs().max()))
        if G > 0:
            theta = theta - math.sqrt(2.0 * math.log(a.K) / (G * G * a.T_eg)) * g
        lam = torch.softmax(theta, 0)
        W = W0 + (lam @ DW).view_as(W0); b = b0 + lam @ DB                # gop ca W va b
        t2 = time.perf_counter()
        if i >= a.khoi_dong:
            tq.append(t1 - t0); tn.append(t2 - t1); tr.append(r_)
            kich_hoat = max(kich_hoat, dem.duy_nhat); tho = max(tho, dem.tho); tam = max(tam, nb(gW, gb, g, lam, W, b))

    # ---- E6, pha khoi dong (cham bac nhat S nguon)
    tq6, tn6, kh6, tam6 = [], [], 0, 0
    acc = torch.zeros(a.S)
    for i in range(a.khoi_dong + a.n):
        if i == a.khoi_dong:
            acc.zero_()                                                  # bat dau lai sau khoi dong may
        x = X[i]; dem.reset()
        t0 = time.perf_counter()
        Wl, bl = W0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(dem.goi, lambda t: t):
            with torch.no_grad():
                h1 = m.l1(x)
            lg = m.sau(h1, Wl, bl); int(lg.argmax(1))
            t1 = time.perf_counter()
            c_ = do_tin(lg).sum()
        gW, gb = torch.autograd.grad(c_, [Wl, bl])
        dong = DWs @ gW.flatten() + DBs @ gb
        acc += dong
        t2 = time.perf_counter()
        if i >= a.khoi_dong:
            tq6.append(t1 - t0); tn6.append(t2 - t1); kh6 = max(kh6, dem.duy_nhat); tam6 = max(tam6, nb(gW, gb, dong))
    t40 = []
    for i in range(a.khoi_dong + a.n):                                   # luot 40: chon 5 nguon + gop W va b
        t0 = time.perf_counter()
        top = torch.argsort(-acc, stable=True)[:a.K]
        Wc = W0 + DWs[top].mean(0).view_as(W0); bc = b0 + DBs[top].mean(0)
        t1 = time.perf_counter()
        if i >= a.khoi_dong:
            t40.append(t1 - t0); tam6 = max(tam6, nb(top, Wc, bc))

    n2 = ns // 2
    mac = dict(l1=3 * C * ns, l2=nch * C * ns, l3=nch * n2 * nch * 2, l4=nch * n2 * nch * 10, fc=nch * n2 * a.N)
    mac_thuan = sum(mac.values()); mac_nguoc = mac["fc"] + mac["l4"] + mac["l3"] + mac["l2"]
    trang_thai = nb(W0, b0, DW, DB, theta)
    print(json.dumps(dict(ban=3, T=a.T, N=a.N, K=a.K, S=a.S, luong=a.luong, seed=a.seed, n=a.n, khoi_dong=a.khoi_dong, torch=torch.__version__,
                          E4=dict(quyet_dinh=q(tq), nen=q(tn), tong=q(np.array(tq) + np.array(tn)), suy_luan_no_grad=q(tr)),
                          E6=dict(quyet_dinh=q(tq6), nen=q(tn6), luot40=q(t40)),
                          mac=dict(suy_luan=mac_thuan, nen_E4=mac_nguoc + 2 * a.K * (nch * C + nch), nen_E6=mac_nguoc + a.S * (nch * C + nch)),
                          byte=dict(kich_hoat_duy_nhat_E4=kich_hoat, kich_hoat_duy_nhat_E6=kh6, tho_moi_lan_luu_E4=tho, tam_E4=tam, tam_E6=tam6,
                                    trang_thai_E4=trang_thai, kho_S=nb(DWs, DBs, acc),
                                    them_E4=kich_hoat + tam + trang_thai, them_E6=kh6 + tam6 + nb(DWs, DBs, acc) + nb(W0, b0)),
                          vmhwm_kb=vmhwm_kb())))


if __name__ == "__main__":
    main()
