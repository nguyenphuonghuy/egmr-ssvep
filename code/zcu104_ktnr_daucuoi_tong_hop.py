#!/usr/bin/env python
"""Tong hop do dau-cuoi KTN-R (cung cach voi zcu104_ktnr_tong_hop.py: TB giua 7 tien trinh cua TB moi tien trinh, p95 lon nhat).
Dung: python zcu104_ktnr_daucuoi_tong_hop.py <jsonl> <json ra>
"""
import json, sys
import numpy as np

R = {}
for d in (json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()):
    R.setdefault((d["T"], d["luong"]), []).append(d)
kq = {}
for (T, L), r in sorted(R.items()):
    e = dict(n=len(r), du=len(r) == 7, trong_so_nen_that=all(x["trong_so_nen_that"] for x in r))
    for k in ("bank_loc", "loi", "tong"):
        e[k + "_tb"] = float(np.mean([x[k]["tb_ms"] for x in r])); e[k + "_sd"] = float(np.std([x[k]["tb_ms"] for x in r], ddof=1))
        e[k + "_p95_max"] = float(max(x[k]["p95_ms"] for x in r)); e[k + "_max"] = float(max(x[k]["max_ms"] for x in r))
    kq["T%s_L%d" % (T, L)] = e
    print("T=%s L=%d n=%d | bank %.2f | loi %.2f | tong %.2f +- %.2f p95max %.2f max %.2f" % (
        T, L, e["n"], e["bank_loc_tb"], e["loi_tb"], e["tong_tb"], e["tong_sd"], e["tong_p95_max"], e["tong_max"]))
json.dump(dict(cau_hinh=kq), open(sys.argv[2], "w", encoding="utf-8"), indent=1)
