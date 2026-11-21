#!/usr/bin/env python
"""HB3: dem MAC MOI QUYET DINH cua KTN-R tren MOT HAM QUYET DINH HOAN CHINH (ban 2).
Ham quyet_dinh(): u = l1(x) goi MOT lan va dung chung cho mang (l2..fc) va nhanh R; mo-dun ca nhan da gop vao l2 (dang trien khai).
Dem:
  Conv2d/Linear : moc CONG DON qua moi lan goi: (phan tu dau ra / lo) x in_ch/groups x kh x kw; Linear: in x out (khong tinh cong bias).
  nhanh R       : bo dem tuong minh tang o tung phep nhan/chia thuc thi: y = w^T u (C ns); |y|^2 (ns); Q_c^T y (2H ns moi lop); |Q_c^T y|^2 (2H moi lop);
                  chia lay ti le (N); z hai vecto: phuong sai (N) + chia (N) moi vecto; nhan alpha (N). Phep cong/tru, can, so sanh khong tinh.
  Q_c (sin/cos QR) tinh SAN mot lan cho moi cua so (hang so luu bo nho, khong tinh vao moi quyet dinh); bank loc ngoai tran (tac gia 2026-11-12), bao rieng.
Kiem: quyet dinh cua ham bang quyet dinh tinh truc tiep theo cong thuc cua ktnr_buoc1.py tren cung dau vao.
"""
import json, sys, os, hashlib
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import global_dnn as GD   # noqa: E402

H, ALPHA = 5, 0.5
torch.manual_seed(20261107)


def co_so(freqs, fs, ns):
    t = np.arange(ns) / fs
    return np.stack([np.linalg.qr(np.stack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)], 1))[0] for f in freqs])


def dem(C, fs, ns, freqs):
    N = len(freqs)
    m = GD.GlobalDNN(C, ns, nclass=N, nch=22).eval()
    with torch.no_grad():
        m.l2.weight += 0.01 * torch.randn_like(m.l2.weight); m.l2.bias += 0.01 * torch.randn_like(m.l2.bias)
    tong = {}

    def moc(ten):
        def f(mod, inp, outp):
            if isinstance(mod, nn.Conv2d):
                v = outp.numel() // outp.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            else:
                v = mod.in_features * mod.out_features
            tong[ten] = tong.get(ten, 0) + int(v)                                   # cong don
        return f
    for ten, mod in m.named_children():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.register_forward_hook(moc(ten))
    Qb = co_so(freqs, fs, ns)                                                       # tinh san
    w = np.random.default_rng(0).standard_normal(C); w /= np.linalg.norm(w)
    r = {"chieu_khong_gian": 0, "chuan_y": 0, "chieu_co_so": 0, "chuan_chieu": 0, "ti_le": 0, "z": 0, "gop": 0}

    def zc(v):
        mu = v.mean(); d = v - mu
        var = float(d @ d) / len(v); r["z"] += len(v)                                # phuong sai: N phep nhan
        out = d / (np.sqrt(var) + 1e-12); r["z"] += len(v)                          # chia: N
        return out

    def quyet_dinh(x):
        with torch.no_grad():
            u_t = m.l1(x)                                                           # MOT lan
            h = m.l2(u_t); h = m.l4(torch.relu(m.l3(h))); lg = m.fc(h.flatten(1))[0].double().numpy()
        u = u_t[0, 0].double().numpy(); u = u - u.mean(-1, keepdims=True)
        y = w @ u; r["chieu_khong_gian"] += C * ns
        ny2 = float(y @ y); r["chuan_y"] += ns
        R = np.empty(N)
        for c in range(N):
            p = Qb[c].T @ y; r["chieu_co_so"] += 2 * H * ns
            R[c] = float(p @ p); r["chuan_chieu"] += 2 * H
        R = np.sqrt(R / ny2) if ny2 > 0 else np.zeros(N); r["ti_le"] += N
        s = zc(lg) + ALPHA * zc(R); r["gop"] += N
        return int(np.argmax(s)), lg, R

    x = torch.randn(1, GD.NSB, C, ns)
    d, lg, R = quyet_dinh(x)
    zz = lambda v: (v - v.mean()) / (v.std() + 1e-12)                               # cong thuc ktnr_buoc1.z
    assert d == int(np.argmax(zz(lg) + ALPHA * zz(R))) and np.isfinite(R).all() and (R <= 1 + 1e-9).all()
    mang = sum(tong.values())
    du_toan = GD.dem_mac(None, C, ns, nch=22); du_toan_mang = du_toan["tong"] - du_toan["fc"] + 22 * (ns // 2) * N
    assert mang == du_toan_mang, (mang, du_toan_mang)
    return dict(theo_lop=tong, mang=mang, nhanh_R=dict(r), tong=mang + sum(r.values()))


out = {}
CFG = (("40lop_9kenh_250Hz", 9, 250, [8 + 0.2 * k for k in range(40)]), ("9lop_9kenh_250Hz", 9, 250, [8.0, 9.5, 11.0, 8.5, 10.0, 11.5, 9.0, 10.5, 12.0]),
       ("12lop_8kenh_256Hz", 8, 256, [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]))
for ten, C, fs, fr in CFG:
    for T in (0.6, 1.0):
        ns = int(round(T * fs)); k_ = "%s_T%.1f" % (ten, T)
        out[k_] = dem(C, fs, ns, fr)
        print(ten, T, out[k_]["tong"], flush=True)
k = out["40lop_9kenh_250Hz_T1.0"]["tong"]
out["HB3"] = dict(gia_tri=k)
out["script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
print("HB3:", out["HB3"])
if len(sys.argv) > 1:
    if os.path.exists(sys.argv[1]):
        sys.exit("tep ket qua da co: khong ghi de")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
