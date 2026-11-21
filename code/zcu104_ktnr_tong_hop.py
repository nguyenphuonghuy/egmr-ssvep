#!/usr/bin/env python
"""Tong hop do KTN-R tren ZCU104..
Dung: python zcu104_ktnr_tong_hop.py <ktnr jsonl> <ktn jsonl 15/11> <json ra>
Moi cau hinh (S, N, T, luong): TB +- SD giua tien trinh cua TB moi tien trinh; p95 lon nhat; can du 7 tien trinh.
"""
import json, sys
import numpy as np

KR, KN, RA = sys.argv[1:4]


def doc(f):
    out = {}
    for d in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip()):
        out.setdefault((d["S"], d["N"], d["T"], d["luong"]), []).append(d)
    return out


R, K = doc(KR), doc(KN)
kq = {}
for key in sorted(R):
    S, N, T, L = key
    r, k = R[key], K.get(key, [])
    ten = "S%d_N%d_T%s_L%d" % key
    du = len(r) == 7
    m = lambda xs: (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)
    sl_tb, sl_sd = m([d["suy_luan"]["tb_ms"] for d in r])
    sl_p95 = max(d["suy_luan"]["p95_ms"] for d in r)
    R_tb, _ = m([d["nhanh_R_va_gop"]["tb_ms"] for d in r])
    kt_tb, kt_sd = m([d["khoi_tao"]["tb_ms"] for d in r])
    kt_p95 = max(d["khoi_tao"]["p95_ms"] for d in r)
    l40 = max(d["luot_40_ms"] for d in r)
    wR = max(d["w_R_ms"] for d in r)
    qb, _ = m([d["dung_Qb_ngoai_tuyen"]["tb_ms"] for d in r])
    e = dict(du_7_tien_trinh=du, n=len(r), suy_luan_tb=sl_tb, suy_luan_sd=sl_sd, suy_luan_p95_max=sl_p95, nhanh_R_tb=R_tb,
             khoi_tao_tb=kt_tb, khoi_tao_sd=kt_sd, khoi_tao_p95_max=kt_p95, luot_40_max=l40, w_R_max=wR, Qb_ngoai_tuyen_tb=qb,
             mac_suy_luan=r[0]["mac_suy_luan"])
    if k:
        e["ktn_suy_luan_tb"] = float(np.mean([d["suy_luan"]["tb_ms"] for d in k]))
        e["ktn_khoi_tao_tb"] = float(np.mean([d["khoi_tao"]["tb_ms"] for d in k]))
        e["them_suy_luan_ms"] = sl_tb - e["ktn_suy_luan_tb"]
        e["khoi_tao_lech_tuong_doi"] = kt_tb / e["ktn_khoi_tao_tb"] - 1.0
    kq[ten] = e
json.dump(dict(cau_hinh=kq),
          open(RA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for t, e in kq.items():
    print(t, "n=%d SL %.2f+-%.2f p95max %.2f | R %.2f | them %.2f | KT %.1f p95max %.1f l40max %.1f wR %.2f Qb %.1f" % (
        e["n"], e["suy_luan_tb"], e["suy_luan_sd"], e["suy_luan_p95_max"], e["nhanh_R_tb"], e.get("them_suy_luan_ms", float("nan")),
        e["khoi_tao_tb"], e["khoi_tao_p95_max"], e["luot_40_max"], e["w_R_max"], e["Qb_ngoai_tuyen_tb"]))
