#!/usr/bin/env python
"""Kiem don vi chon_dong (ensemble_dnn.py) so voi vong lap thuan theo testing.m (MATLAB): sort giam dan, cong phieu co trong so,
le = top1 - top2, max lay k dau tien, quyet dinh argmax cua phieu tai k chon. Khong dung du lieu EEG."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_dnn as E


def tham_chieu(sim, pred, NT):
    S, n = sim.shape
    kq = np.zeros(n, int); kc = np.zeros(n, int)
    for i in range(n):
        od = np.argsort(-sim[:, i], kind="stable")
        s, p = sim[od, i], pred[od, i]
        le_k, dec_k = [], []
        for k in range(1, S + 1):
            ph = np.zeros(NT)
            for j in range(k):
                ph[p[j]] += s[j]
            t = np.sort(ph)[::-1]
            le_k.append(t[0] - t[1]); dec_k.append(int(np.argmax(ph)))
        k = int(np.argmax(le_k))
        kq[i] = dec_k[k]; kc[i] = k + 1
    return kq, kc


rng = np.random.default_rng(20261107)
for NT, S, n in ((40, 34, 200), (40, 69, 100), (9, 99, 100)):
    E.GD.NT = NT
    sim = rng.random((S, n)).astype(np.float32)
    pred = rng.integers(0, NT, (S, n))
    pred[: S // 2] = pred[0]                       # nhieu nguon cung du doan
    a, b = E.chon_dong(torch.tensor(sim), torch.tensor(pred))
    r, c = tham_chieu(sim, pred, NT)
    assert (a.numpy() == r).all() and (b.numpy() == c).all(), (NT, S, (a.numpy() != r).sum(), (b.numpy() != c).sum())
    print("khop", NT, S, n)
print("KIEM CHON_DONG DAT")
