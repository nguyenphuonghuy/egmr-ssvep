#!/usr/bin/env python
"""Tong hop THI NGHIEM B.
Cau hinh quyet dinh: T = 1,0 s, 1 luong. B1, B2: p95 tung tien trinh, quyet dinh theo p95 LON NHAT qua 7 tien trinh (nhu cong R); trung vi bao TB qua tien trinh.
B2 nhanh: p95 lon nhat qua tien trinh cua tung nhanh; bao gia tri lon hon; nhanh vang ghi "khong co mau".
B3: moc tron ven (do) theo fold, a; vong tong hop (workload tong hop); du toan a = 220 = trung vi mot vong (a = 220) x 68,3 vong (TB GPU, ghi la du toan).
Cau tuyen bo (chi gan voi ca do: nguoi 1 Benchmark P1 r = 1 cho B2; fold 17, 27 cho B3).
Du so do: moi cau hinh B1/B2 phai co du tien trinh 1..7, moi tien trinh dung mot dong, ma thoat 0 trong trang_thai.tsv,
so luot dung (B1: 1.000; B2: 5 phien x 240 = 1.200); thieu thi ghi "chua du so do" va KHONG phat cau tuyen bo cho cau hinh do.
B3 moc: moi ca (fold 17, 27 x a 40, 220) phai co dong hoac ma thoat; ma 124 (timeout cua shell) = "> 30 phut, chua hoan tat"; ma khac 0 = loi, giu.
Dung: python zcu104_doi_thu_tong_hop.py <thu muc phien_...> <json ra>
"""
import json, sys
import numpy as np

VONG_TB_A220 = 68.342857                     # TB tong so vong qua sau lam o a = 220, Benchmark 1,0 s (d6/sfda_tt_benchmark_T10.json)


def main():
    import os
    D = sys.argv[1]
    dong = [json.loads(l) for l in open(os.path.join(D, "doi_thu.jsonl"), encoding="utf-8") if l.strip().startswith("{")]
    tt_ = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(D, "trang_thai.tsv"), encoding="utf-8") if l.strip())
    out = dict(cau=[], trang_thai_ca=tt_, du={})
    def du(phan, T, L, n_luot):
        ds = [d for d in dong if d["phan"] == phan and abs(d["T"] - T) < 1e-9 and d["luong"] == L]
        ids = sorted(d["tien_trinh"] for d in ds)
        rc = [tt_.get(f"{phan} T={T} L={L} r={r}") for r in range(1, 8)]
        ok_n = all(d["ket_qua"]["tong_cung_luot"]["n"] == n_luot for d in ds)
        return ids == list(range(1, 8)) and all(v == "0" for v in rc) and ok_n
    def gom(phan, T, L, duong):
        ds = [d for d in dong if d["phan"] == phan and abs(d["T"] - T) < 1e-9 and d["luong"] == L]
        v = []
        for d in ds:
            x = d["ket_qua"]
            for k in duong:
                x = x[k]
            v.append(x)
        return v
    for T, L in ((1.0, 1), (1.0, 4), (0.6, 1)):
        key = f"T{T}_L{L}"
        out["du"][key] = dict(B1=du("b1", T, L, 1000), B2=du("b2", T, L, 1200))
        b1 = {k: gom("b1", T, L, [k]) for k in ("bank_loc", "quyet_dinh_tu_tho", "nen", "tong_cung_luot")}
        b2 = {k: gom("b2", T, L, [k]) for k in ("quyet_dinh_tu_tho", "quyet_dinh_loi", "cap_nhat", "tong_cung_luot", "tien_xu_ly_cache")}
        tt = lambda v: dict(so_tien_trinh=len(v), p95_lon_nhat_ms=max(x["p95_ms"] for x in v), trung_vi_tb_ms=float(np.mean([x["trung_vi_ms"] for x in v])),
                            vuot_50ms_tb=float(np.mean([x["vuot_50ms"] for x in v])), vuot_500ms_tb=float(np.mean([x["vuot_500ms"] for x in v]))) if v else None
        o = dict(B1={k: tt(v) for k, v in b1.items()}, B2={k: tt(v) for k, v in b2.items()})
        nh = {}
        for ten in ("res1_bang_res6", "res1_khac_res6"):
            ds = gom("b2", T, L, ["theo_nhanh", ten])
            co = [x for x in ds if isinstance(x["cap_nhat"], dict)]
            nh[ten] = dict(so_mau=int(sum(x["so_mau"] for x in ds)),
                           cap_nhat_p95_lon_nhat_ms=(max(x["cap_nhat"]["p95_ms"] for x in co) if co else "khong co mau"),
                           tong_p95_lon_nhat_ms=(max(x["tong_cung_luot"]["p95_ms"] for x in co) if co else "khong co mau"))
        o["B2_nhanh"] = nh
        out[key] = o
    q = out["T1.0_L1"]
    if not out["du"]["T1.0_L1"]["B2"]:
        out["cau"].append("B2 (1,0 s, 1 luong): chua du so do (thieu tien trinh, ma thoat khac 0 hoac sai so luot); khong phat cau tuyen bo")
    elif q["B2"]["cap_nhat"]:
        cn = max([q["B2"]["cap_nhat"]["p95_lon_nhat_ms"]] + [v["cap_nhat_p95_lon_nhat_ms"] for v in q["B2_nhanh"].values() if not isinstance(v["cap_nhat_p95_lon_nhat_ms"], str)])
        tg = max([q["B2"]["tong_cung_luot"]["p95_lon_nhat_ms"]] + [v["tong_p95_lon_nhat_ms"] for v in q["B2_nhanh"].values() if not isinstance(v["tong_p95_lon_nhat_ms"], str)])
        ca = "(nguoi 1 Benchmark, P1 r = 1, 1,0 s, 1 luong)"
        out["cau"].append(f"B2 {ca}: " + ("phan cap nhat cua OACCA-NQ nam trong nguong nen 50 ms" if cn < 50 else "phan cap nhat cua OACCA-NQ vuot nguong nen 50 ms (R2)") + f" (p95 {cn:.1f} ms)")
        if tg < 500:
            out["cau"].append(f"B2 {ca}: OACCA-NQ chay kip trong khoang nghi 500 ms tren PS (p95 tong {tg:.1f} ms); khong duoc noi E4 la cach kha thi duy nhat")
        if cn >= 50:
            out["cau"].append("B2: neu E4 dat cong R: 'trong cac doi thu da do, chi E4 dat nguong nen 50 ms'")
    if not out["du"]["T1.0_L1"]["B1"]:
        out["cau"].append("B1 (1,0 s, 1 luong): chua du so do")
    moc = [d["ket_qua"] for d in dong if d["phan"] == "b3moc"]
    out["B3_moc"] = moc
    for f_ in (17, 27):                                                   # moi ca doi chieu so ban ghi, ma thoat, trang thai
        for a_ in (40, 220):
            ten = f"fold {f_}, a = {a_}"
            rc = tt_.get(f"b3moc a={a_} f={f_}")
            bg = [m for m in moc if m["fold"] == f_ and m["a"] == a_]
            if rc == "124" and not bg:                                    # timeout toan tien trinh, khong suy thoi gian moc
                out["cau"].append(f"B3 {ten}: timeout toan tien trinh (2100 s), chua co thoi gian moc hop le; khong phat cau R2")
                continue
            if rc != "0" or len(bg) != 1:
                out["cau"].append(f"B3 {ten}: KHONG HOP LE (ma thoat {rc}, {len(bg)} ban ghi); khong phat cau R2; xem loi.log")
                continue
            m = bg[0]
            if m["moc"]["trang_thai"] != "hoan tat":
                out["cau"].append(f"B3 {ten}: moc SFDA-TT > 30 phut, chua hoan tat; vuot khoang nghi 500 ms va nguong nen 50 ms")
            elif m["moc"]["tg_s"] > 0.5:
                out["cau"].append(f"B3 {ten}: tren ca do, mot moc SFDA-TT keo dai {m['moc']['tg_s']:.1f} s ({m['moc']['tong_vong']} vong), vuot khoang nghi 500 ms")
            elif m["moc"]["tg_s"] > 0.05:
                out["cau"].append(f"B3 {ten}: moc {1e3 * m['moc']['tg_s']:.0f} ms, vuot nguong nen 50 ms")
            else:
                out["cau"].append(f"B3 {ten}: moc <= 50 ms: bo cau vuot R2 cua SFDA-TT")
    bn = []
    for f_ in (17, 27):
        for a_ in (40, 220):
            rc = tt_.get(f"b3bonho a={a_} f={f_}")
            bg = [d["ket_qua"] for d in dong if d["phan"] == "b3bonho" and d["ket_qua"]["fold"] == f_ and d["ket_qua"]["a"] == a_]
            if rc != "0" or len(bg) != 1:
                out["cau"].append(f"B3 bo nho fold {f_}, a = {a_}: KHONG HOP LE (ma thoat {rc}, {len(bg)} ban ghi)")
                continue
            b_ = bg[0]; bn.append(b_)
            if b_["dau_vao_X_byte"] > 1_000_000:
                out["cau"].append(f"B3 bo nho fold {f_}, a = {a_}: rieng dau vao SFDA-TT phai giu da {b_['dau_vao_X_byte']} byte (can duoi chinh xac), vuot R3 (1 MB)")
    out["B3_bo_nho"] = bn
    vong = {d["ket_qua"]["a"]: d["ket_qua"] for d in dong if d["phan"] == "b3vong"}
    out["B3_vong_tong_hop"] = vong
    out["B3_phu"] = [d["ket_qua"] for d in dong if d["phan"] == "b3phu"]
    if 220 in vong:
        out["B3_du_toan_moc_a220_s"] = dict(gia_tri=vong[220]["trung_vi_s"] * VONG_TB_A220, ghi_chu="DU TOAN: trung vi mot vong tong hop x 68,3 vong TB GPU; chua gom tau_phu")
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for c in out["cau"]:
        print(c)


if __name__ == "__main__":
    main()
