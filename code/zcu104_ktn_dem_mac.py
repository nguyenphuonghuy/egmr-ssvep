#!/usr/bin/env python
"""Dem MAC THUC THI cua duong quyet dinh KTN bang moc (hook) tren tung lop Conv2d/Linear cua chinh mang chay tren board (H4, ve MAC)."""
import json, torch, torch.nn as nn
from zcu104_ktn_dotre import GlobalDNN
torch.manual_seed(20261107)
out = {}
for N in (40, 9):
    for T in (0.6, 1.0):
        ns = int(round(T * 250)); m = GlobalDNN(9, ns, N).eval(); tong = {}
        def moc(ten):
            def f(mod, inp, outp):
                if isinstance(mod, nn.Conv2d):
                    k = mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
                    tong[ten] = int(outp.numel() * k)
                else:
                    tong[ten] = int(mod.in_features * mod.out_features)
            return f
        for ten, mod in m.named_children():
            mod.register_forward_hook(moc(ten))
        with torch.no_grad():
            m(torch.randn(1, 3, 9, ns))
        # l2 trong duong quyet dinh goi qua Fn.conv2d (mo-dun da gop): tinh bang cung cong thuc tren dau ra that
        with torch.no_grad():
            h = torch.nn.functional.conv2d(m.l1(torch.randn(1, 3, 9, ns)), m.l2.weight, m.l2.bias)
        tong["l2"] = int(h.numel() * 1 * 9 * 1)
        out["N%d_T%.1f" % (N, T)] = dict(theo_lop=tong, tong=sum(tong.values()))
print(json.dumps(out))
