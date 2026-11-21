#!/usr/bin/env python
"""Dem MAC MOI QUYET DINH cua Ensemble-DNN (ban tai hien ensemble_dnn.py, theo testing.m cua ma goc) cung quy uoc voi ktnr_dem_mac.py.
Mot quyet dinh = voi MOI nguon s (S = so nguoi - 1): forward toan mang nguon s + do giong (do_giong trong ensemble_dnn.py) + bo phieu.
Quy uoc (thuan loi nhat cho Ensemble-DNN, de khong thoi phong khoang cach):
  - u = l1(x) va W x (l2 chua bias) dung lai tu forward; tru TB lam mat bias nen zx = l2(u) - TB.
  - mau lop da chieu qua bo loc (zt, da tru TB) va |zt|^2 tinh SAN ngoai tuyen cho moi nguon, moi lop (luu bo nho, khong tinh).
  - Moi nguon them: <zt, zx> (K ns), |zx|^2 (K ns), chia + nhan trong mau so (2K), r^2 (K); CCA: zx Q (2H K ns), binh phuong tong (2H K),
    chia (K). Bo phieu: cong, so sanh (khong tinh). Phep cong/tru, can, so sanh khong tinh (nhu ktnr_dem_mac.py).
  - Bank loc ngoai tran, khong tinh (nhu KTN-R).
Kiem: so dem bang moc tren mang = cong thuc GD.dem_mac cua cung mang.
"""
import json, os, sys, hashlib
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import global_dnn as GD   # noqa: E402

H = 5


def mang_mac(C, ns, N, nch):
    m = GD.GlobalDNN(C, ns, nclass=N, nch=nch).eval()
    tong = {}

    def moc(ten):
        def f(mod, inp, outp):
            if isinstance(mod, nn.Conv2d):
                v = outp.numel() // outp.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
            else:
                v = mod.in_features * mod.out_features
            tong[ten] = tong.get(ten, 0) + int(v)
        return f
    for ten, mod in m.named_children():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.register_forward_hook(moc(ten))
    with torch.no_grad():
        m(torch.randn(1, GD.NSB, C, ns))
    s = sum(tong.values())
    du = GD.dem_mac(None, C, ns, nch=nch); du_toan = du["tong"] - du["fc"] + nch * (ns // 2) * N
    assert s == du_toan, (s, du_toan)
    return s, tong


def do_giong_mac(K, ns):
    return dict(tich_zt_zx=K * ns, chuan_zx=K * ns, mau_so=2 * K, r2=K, chieu_co_so=2 * H * K * ns, chuan_chieu=2 * H * K, chia_cca=K)


out = {}
for ds, S in (("benchmark", 34), ("beta", 69)):
    for nch, ten in ((22, "ED_22"), (120, "ED_du")):
        for T in (0.6, 1.0):
            ns = int(round(T * 250)); N = 40
            mg, theo_lop = mang_mac(9, ns, N, nch)
            dg = do_giong_mac(nch, ns)
            moi_nguon = mg + sum(dg.values())
            k = "%s_%s_T%.1f" % (ds, ten, T)
            out[k] = dict(S=S, nch=nch, mang_moi_nguon=mg, theo_lop=theo_lop, do_giong_moi_nguon=dg, moi_nguon=moi_nguon, tong=S * moi_nguon)
            print(k, "S=%d moi nguon %d tong %d" % (S, moi_nguon, S * moi_nguon), flush=True)
out["quy_uoc"] = __doc__
out["script_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
if len(sys.argv) > 1:
    if os.path.exists(sys.argv[1]):
        sys.exit("tep ket qua da co: khong ghi de")
    json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
