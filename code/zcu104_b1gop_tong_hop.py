#!/usr/bin/env python
"""Tong hop cong R tu b1gop_eg.jsonl cua sweep ZCU104.
Cau hinh quyet dinh: PyTorch, 1 luong, T = 1,0 s (Benchmark, 40 lop), 1.000 luot sau 20 luot khoi dong may.
Cach gop qua 7 tien trinh doc lap (chon cach than trong nhat): R dat neu
  max qua tien trinh cua p95 thoi gian nen E4 < 50 ms  VA  MAC nen E4 <= 1.000.000  VA  max qua tien trinh cua byte them E4 <= 1 MB (1.048.576).
Bao kem (khong la dieu kien): 4 luong, 0,6 s, trung vi/TB +- do lech chuan qua tien trinh, E6 (nen moi luot, luot 40), suy luan, VmHWM.
Dung: python zcu104_b1gop_tong_hop.py <b1gop_eg.jsonl> <json ra>"""
import json, sys
import numpy as np

r = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
assert all(x["ban"] == 3 and x["n"] == 1000 and x["khoi_dong"] == 20 for x in r), "sai ban ma hoac cau hinh lay mau"
out = {"cau_hinh": {}, "cong_R": {}}
for T in (1.0, 0.6):
    for L in (1, 4):
        g = [x for x in r if x["T"] == T and x["luong"] == L]
        if not g:
            continue
        f = lambda k1, k2, k3: [x[k1][k2][k3] for x in g]
        tk = lambda v: dict(tb=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, max=float(np.max(v)))
        out["cau_hinh"][f"T{T}_L{L}"] = dict(
            so_tien_trinh=len(g), seed=[x["seed"] for x in g],
            E4_nen_trung_vi=tk(f("E4", "nen", "trung_vi_ms")), E4_nen_p95=tk(f("E4", "nen", "p95_ms")), E4_nen_max=tk(f("E4", "nen", "max_ms")),
            E4_quyet_dinh_trung_vi=tk(f("E4", "quyet_dinh", "trung_vi_ms")), E4_tong_p95=tk(f("E4", "tong", "p95_ms")),
            suy_luan_no_grad_trung_vi=tk(f("E4", "suy_luan_no_grad", "trung_vi_ms")),
            E6_nen_p95=tk(f("E6", "nen", "p95_ms")), E6_luot40_p95=tk(f("E6", "luot40", "p95_ms")),
            mac=g[0]["mac"], byte_them_E4_max=int(max(x["byte"]["them_E4"] for x in g)), byte_them_E6_max=int(max(x["byte"]["them_E6"] for x in g)),
            vmhwm_kb_max=int(max(x["vmhwm_kb"] for x in g)))
q = out["cau_hinh"].get("T1.0_L1")
assert q and q["so_tien_trinh"] == 7, "thieu cau hinh quyet dinh"
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in out["cau_hinh"].items():
    print(k, "nen E4 trung vi %.2f p95 max %.2f ms | quyet dinh %.2f | suy luan %.2f | E6 nen p95 %.2f | luot40 p95 %.2f | byte E4 %d E6 %d | VmHWM %d kB" % (
        v["E4_nen_trung_vi"]["tb"], v["E4_nen_p95"]["max"], v["E4_quyet_dinh_trung_vi"]["tb"], v["suy_luan_no_grad_trung_vi"]["tb"],
        v["E6_nen_p95"]["max"], v["E6_luot40_p95"]["max"], v["byte_them_E4_max"], v["byte_them_E6_max"], v["vmhwm_kb_max"]))
