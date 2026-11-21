#!/usr/bin/env python
"""Tong hop do E4R-b tren ZCU104.
Moi cau hinh can du tien trinh 1..7, moi tien trinh mot dong, ma thoat 0, n = 1.000; thieu thi "chua du so do", khong phat cong.
Dung: python zcu104_b2e4r_tong_hop.py <thu muc phien> <json ra>
"""
import json, os, sys
import numpy as np


def main():
    D, ra = sys.argv[1], sys.argv[2]
    assert not os.path.exists(ra), "tep ra da co: khong ghi de"
    dong = [json.loads(l) for l in open(os.path.join(D, "b2e4r.jsonl"), encoding="utf-8") if l.strip().startswith("{")]
    tt = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(D, "trang_thai.tsv"), encoding="utf-8") if l.strip())
    out = dict(trang_thai=tt, cau_hinh={})
    for T, L in ((1.0, 1), (1.0, 4), (0.6, 1)):
        ds = [d for d in dong if abs(d["T"] - T) < 1e-9 and d["luong"] == L]
        du = (sorted(d["tien_trinh"] for d in ds) == list(range(1, 8)) and all(tt.get(f"T={T} L={L} r={r}") == "0" for r in range(1, 8))
              and all(d["ket_qua"]["nen"]["n"] == 1000 for d in ds))
        o = dict(du=du, so_tien_trinh=len(ds))
        for k in ("bank_loc", "nhanh_R", "quyet_dinh_tu_tho", "nen", "tong_cung_luot", "mot_lan_moi_phien"):
            v = [d["ket_qua"][k] for d in ds]
            o[k] = dict(trung_vi_tb_ms=float(np.mean([x["trung_vi_ms"] for x in v])), p95_lon_nhat_ms=float(max(x["p95_ms"] for x in v)),
                        vuot_50ms_tb=float(np.mean([x["vuot_50ms"] for x in v])), vuot_500ms_tb=float(np.mean([x["vuot_500ms"] for x in v]))) if v else None
        if ds:
            o["mac"] = ds[0]["mac"]
            o["byte_lon_nhat"] = {k: max(d["byte"][k] for d in ds) for k in ds[0]["byte"]}
            o["vmhwm_kb_lon_nhat"] = max(d["vmhwm_kb"] for d in ds)
        out["cau_hinh"][f"T{T}_L{L}"] = o
    with open(ra, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, o in out["cau_hinh"].items():
        if o.get("nen"):
            print(k, "du" if o["du"] else "THIEU", {x: (round(o[x]["trung_vi_tb_ms"], 2), round(o[x]["p95_lon_nhat_ms"], 2)) for x in ("bank_loc", "nhanh_R", "quyet_dinh_tu_tho", "nen", "tong_cung_luot", "mot_lan_moi_phien")})
    print("byte", out["cau_hinh"]["T1.0_L1"].get("byte_lon_nhat"))


if __name__ == "__main__":
    main()
