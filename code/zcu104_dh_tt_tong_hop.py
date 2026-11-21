#!/usr/bin/env python
"""Tong hop do DH-TT tren ZCU104 (1,0 s, 1 luong). Cach tong hop nhu zcu104_b2e4r_tong_hop.py: moi cau hinh can du tien trinh 1..7, moi tien trinh mot dong,
ma thoat 0, n = 1.000, va lan --kiem dat; thieu thi "chua du so do". p95: lon nhat qua 7 tien trinh; trung vi: TB qua tien trinh.
Bao rieng tung dieu kien (khong gop thanh mot cong): R1 MAC quyet dinh <= 1e6; R2 MAC nen <= 1e6; R3 bo nho them <= 1 MB;
p95 nen < 50 ms. Ket luan R2 o 1,0 s khong keo sang 0,6 s.
Dung: python zcu104_dh_tt_tong_hop.py <thu muc phien> <json ra>
"""
import json, os, sys
import numpy as np


def main():
    D, ra = sys.argv[1], sys.argv[2]
    assert not os.path.exists(ra), "tep ra da co: khong ghi de"
    dong = [json.loads(l) for l in open(os.path.join(D, "dhtt.jsonl"), encoding="utf-8") if l.strip().startswith("{")]
    tt = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(D, "trang_thai.tsv"), encoding="utf-8") if l.strip())
    kiem = json.load(open(os.path.join(D, "kiem.json"), encoding="utf-8"))["kiem"]
    out = dict(trang_thai=tt, kiem=kiem, cau_hinh={}, dieu_kien=None)
    for T, L in ((1.0, 1), (1.0, 4), (0.6, 1)):
        ds = [d for d in dong if abs(d["T"] - T) < 1e-9 and d["luong"] == L]
        du = (tt.get("kiem") == "0" and kiem["dat"] and sorted(d["tien_trinh"] for d in ds) == list(range(1, 8))
              and all(tt.get(f"T={T} L={L} r={r}") == "0" for r in range(1, 8)) and all(d["ket_qua"]["nen"]["n"] == 1000 for d in ds))
        o = dict(du=du, so_tien_trinh=len(ds))
        for k in ("bank_loc", "ea", "xuoi", "quyet_dinh_tu_tho", "nen", "tong_cung_luot"):
            v = [d["ket_qua"][k] for d in ds]
            o[k] = dict(trung_vi_tb_ms=float(np.mean([x["trung_vi_ms"] for x in v])), p95_lon_nhat_ms=float(max(x["p95_ms"] for x in v)),
                        vuot_50ms_tb=float(np.mean([x["vuot_50ms"] for x in v])), vuot_500ms_tb=float(np.mean([x["vuot_500ms"] for x in v]))) if v else None
        if ds:
            o["mac"] = ds[0]["mac"]
            o["byte_lon_nhat"] = {k: max(d["byte"][k] for d in ds) for k in ds[0]["byte"]}
            o["vmhwm_kb_lon_nhat"] = max(d["vmhwm_kb"] for d in ds)
        out["cau_hinh"][f"T{T}_L{L}"] = o
    c = out["cau_hinh"]["T1.0_L1"]
    if not c["du"]:
        out["dieu_kien"] = "CHUA DU SO DO (1,0 s, 1 luong)"
    else:
        out["dieu_kien"] = dict(R1_mac_quyet_dinh_duoi_1M=c["mac"]["quyet_dinh"] <= 1_000_000, R2_mac_nen_duoi_1M=c["mac"]["nen"] <= 1_000_000,
                                R3_bo_nho_them_duoi_1MB=c["byte_lon_nhat"]["them_thich_nghi"] <= 1_000_000,
                                p95_nen_duoi_50ms=c["nen"]["p95_lon_nhat_ms"] < 50)
        do = dict(mac_quyet_dinh=c["mac"]["quyet_dinh"], mac_nen=c["mac"]["nen"], p95_nen_ms=c["nen"]["p95_lon_nhat_ms"],
                  ea_tang_them_ms=c["ea"]["p95_lon_nhat_ms"], bo_nho_them_byte=c["byte_lon_nhat"]["them_thich_nghi"])
        out["ghi_chu"] = "ea_tang_them_ms lay p95 cua thoi gian EA; bo nho them = kich hoat duy nhat + gradient + trang thai EA va F"
    with open(ra, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, o in out["cau_hinh"].items():
        if o.get("nen"):
            print(k, "du" if o["du"] else "THIEU", {x: (round(o[x]["trung_vi_tb_ms"], 2), round(o[x]["p95_lon_nhat_ms"], 2)) for x in ("bank_loc", "ea", "xuoi", "quyet_dinh_tu_tho", "nen", "tong_cung_luot")})
    print("byte", c.get("byte_lon_nhat"), "| dieu kien:", out["dieu_kien"])


if __name__ == "__main__":
    main()
