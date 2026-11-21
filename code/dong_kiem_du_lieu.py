#!/usr/bin/env python
"""Kiem cau truc Dong 2023 TRUOC moi phuong phap giai ma. Khong chay bo giai ma nao.
1. Du 59 tep, eegdata (8, 1250, 40, 4), huu han.
2. Anh xa chi so muc tieu k -> tan so: pho TB moi nguoi, moi khoi o kenh Oz (chi so 5 theo thu tu bai: POz PO3 PO4 PO7 PO8 Oz O1 O2),
   cua so [onset 125 + tre 40, + 3 s), tru TB; diem moi tan so ung vien f = bien do FFT tai f + tai 2f (noi suy theo bin gan nhat, do phan giai 1/3 Hz
   -> dung zero-padding toi 0,02 Hz). Dat khi tan so ung vien diem cao nhat bang 8 + 0,2 k o >= 36/40 chi so.
Ghi JSON: so chi so khop, bang ung vien thang cho moi k. Khong ghi do chinh xac giai ma nao.
"""
import json, os, sys
import numpy as np
from scipy.io import loadmat

D = "/gpfs0/users/gv_dnton/datasets/ssvep/dong2023"
F = 8 + 0.2 * np.arange(40)
X = []
for s in range(1, 60):
    e = loadmat(os.path.join(D, "S%d.mat" % s))["eegdata"]
    assert e.shape == (8, 1250, 40, 4) and np.isfinite(e).all(), (s, e.shape)
    X.append(e[5, 165:165 + 750])                       # Oz, 3 s
X = np.stack(X)                                          # (59, 750, 40, 4)
X = X - X.mean(1, keepdims=True)
m = X.mean((0, 3))                                       # (750, 40): TB qua nguoi va khoi
nfft = 12500                                             # 0,02 Hz
A = np.abs(np.fft.rfft(m * np.hanning(750)[:, None], n=nfft, axis=0))
fr = np.fft.rfftfreq(nfft, 1 / 250)
idx = lambda f: int(round(f / (fr[1] - fr[0])))
diem = np.array([[A[idx(f), k] + A[idx(2 * f), k] for f in F] for k in range(40)])   # (k, ung vien)
thang = F[diem.argmax(1)]
khop = int(np.isclose(thang, F).sum())
out = dict(so_tep=59, khop=khop, nguong=36, dat=bool(khop >= 36), thang=[float(v) for v in thang])
print("anh xa chi so -> 8 + 0,2 k: khop %d/40 ->" % khop, "DAT" if out["dat"] else "KHONG DAT")
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
