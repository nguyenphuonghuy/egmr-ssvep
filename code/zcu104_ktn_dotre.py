#!/usr/bin/env python
"""Do tre KTN tren ZCU104 (PS, Cortex-A53) cho H4. Tu chua, chi can torch + numpy.
Mang nen: GlobalDNN thu nho (giong global_dnn.py: 3 dai con, l2 22 kenh dac trung, l3 1x2 buoc 2, ReLU, l4 1x10 'same', FC), trong so ngau nhien.
Mo-dun nguon s: so gia cua l2 (22 x 9 + 22). Pham vi do:
  luot khoi tao (luot 1..40): quyet dinh bang mang nen + cham S mo-dun (l2..FC cho tung mo-dun, l1 tinh mot lan) + do tin + cong don;
  luot 40 con them: TB tham so 5 mo-dun do tin cao nhat va gop vao l2;
  luot suy luan (sau 40): mot lan forward mang nen + mo-dun ca nhan (gop san vao l2).
Khong gom bank loc va tien xu ly tin hieu tho (do rieng neu can). Batch 1, eval, no_grad. Ghi JSON mot dong ra stdout.
"""
import argparse, json, time, os
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

    def forward(self, x):
        return self.sau(self.l1(x), self.l2.weight, self.l2.bias)


def do_tin(lg):
    s, _ = torch.sort(lg, dim=-1); o = s[..., :-1]
    return (s[..., -1] - o.mean(-1)) / (o.std(-1, unbiased=False) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--luong", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--n-sau", type=int, default=100)
    a = ap.parse_args()
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed)
    C, ns = 9, int(round(a.T * 250))
    m = GlobalDNN(C, ns, a.N).eval()
    dW = [0.01 * torch.randn_like(m.l2.weight) for _ in range(a.S)]; db = [0.01 * torch.randn_like(m.l2.bias) for _ in range(a.S)]
    Ws = [m.l2.weight + d for d in dW]; bs = [m.l2.bias + d for d in db]
    X = torch.randn(40 + a.n_sau, 1, 3, C, ns)
    t_kd, t_sau = [], []
    acc = torch.zeros(a.S)
    with torch.no_grad():
        for _ in range(5):                                                     # khoi dong nong (khong tinh)
            m(X[0]); m.sau(m.l1(X[0]), Ws[0], bs[0])
        for i in range(40):
            t0 = time.perf_counter()
            x = X[i]
            d = int(m(x).argmax(1))                                            # quyet dinh bang mang nen
            h1 = m.l1(x)
            for s in range(a.S):
                acc[s] += do_tin(m.sau(h1, Ws[s], bs[s]))[0]
            if i == 39:
                top = torch.argsort(-acc, stable=True)[:5]
                Wc = m.l2.weight + torch.stack([dW[int(j)] for j in top]).mean(0)
                bc = m.l2.bias + torch.stack([db[int(j)] for j in top]).mean(0)
            t_kd.append(time.perf_counter() - t0)
        for i in range(40, 40 + a.n_sau):
            t0 = time.perf_counter()
            d = int(m.sau(m.l1(X[i]), Wc, bc).argmax(1))
            t_sau.append(time.perf_counter() - t0)
    n2 = ns // 2
    mac_mang = 3 * C * ns + 22 * C * ns + 22 * n2 * 22 * 2 + 22 * n2 * 22 * 10 + 22 * n2 * a.N
    q = lambda v: dict(tb_ms=1e3 * float(np.mean(v)), trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)), max_ms=1e3 * float(np.max(v)))
    print(json.dumps(dict(S=a.S, N=a.N, T=a.T, luong=a.luong, seed=a.seed, torch=torch.__version__, khoi_tao=q(t_kd), luot_40_ms=1e3 * t_kd[-1], suy_luan=q(t_sau),
                          mac_suy_luan=mac_mang, mac_khoi_tao_moi_luot=mac_mang + a.S * (mac_mang - 3 * C * ns))))


if __name__ == "__main__":
    main()
