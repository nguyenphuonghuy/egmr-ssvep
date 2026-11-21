#!/usr/bin/env python
"""Tong hop do thoi gian doi thu B1 tren ZCU104.
 Ban 4.
Dung: python zcu104_doithu_ktnr_tong_hop.py <thu muc phien> <json ra>
Hop le moi cau hinh (pp, T, luong): du 7 dong so ghi cua phien voi r = 1..7, seed = 20261107 + r, ma thoat 0; tep JSON doc duoc, cung phien,
script_sha256 (= ban da qua kiem); n = 200 (Ensemble-DNN ban goc: 20 neu tep tham do cung cau hinh > 1 s), khoi_dong = 20,
tong.n = n, cac thong ke huu han va > 0, luot_diem_hong = 0. Ca 12 tep kiem (6 phuong phap x 2 cua so) phai cung phien, n = 40, so_khac = 0,
luot_diem_hong = 0, dat = True va cung script_sha256 voi phep do.
Quy tac so: 'CSS-R nhanh hon X' khi max cua 7 TB tien trinh CSS-R < min cua 7 TB tien trinh X (cung T, luong); bao ti le hai TB.
'X khong kip khoang nghi 0,5 s' khi p95 lon nhat qua 7 tien trinh > 500 ms.
Cau hinh hoac menh de thieu bang chung: null, khong gop voi 'sai'.
"""
import csv, json, math, os, sys
import numpy as np

PP = ["ktnr", "fbcca", "rcs", "ed22", "edgoc", "ccanet"]; DOI = PP[1:]; TS = ("1.0", "0.6"); LS = ("4", "1")
MAC = {"ktnr": 995390, "fbcca": 4882500, "rcs": 7400000, "ed22": 32600000, "ccanet": 67200000, "edgoc": 777000000}   # 1,0 s, so cua bai (ccanet: can duoi)
PH, RA = sys.argv[1:3]
if os.path.exists(RA):
    sys.exit("tep ket qua da co: khong ghi de")
ten_ph = os.path.basename(os.path.normpath(PH)); goc_dir = os.path.dirname(os.path.normpath(PH))
huu = lambda v: isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def doc(p):
    try:
        j = json.load(open(p))
        return j if isinstance(j, dict) else None
    except Exception:
        return None


HEX = set("0123456789abcdef")
la_sha = lambda s: isinstance(s, str) and len(s) == 64 and set(s) <= HEX
KHOA_MT = ("torch", "numpy", "python", "scipy", "sklearn", "joblib", "threadpoolctl", "sha256")
mt = lambda j: {k: (j.get("moi_truong") or {}).get(k) for k in KHOA_MT}


# ---- 12 tep kiem ----
kiem, sha_kiem = {}, set()
for pp in PP:
    for T in TS:
        j = doc(os.path.join(PH, "kiem", "%s_T%s.json" % (pp, T)))
        ok = bool(j and j.get("dat") is True and j.get("phien") == ten_ph and j.get("pp") == pp and abs(j.get("T", -1) - float(T)) < 1e-9
                  and j.get("n") == 40 and j.get("so_khac") == 0 and j.get("luot_diem_hong") == 0)
        kiem["%s_T%s" % (pp, T)] = ok
        if j:
            sha_kiem.add(j.get("script_sha256"))
# ban ma: hash hop le, duy nhat, khop sha.txt cua phien; moi truong (hash ma phu, du lieu, trong so, ma doi chung; phien ban thu vien) duy nhat
sha_txt = {}
try:
    for l in open(os.path.join(PH, "sha.txt")):
        h, fn = l.split()
        sha_txt[os.path.basename(fn.lstrip("*"))] = h
except Exception:
    pass
MT_REF, mt_ok = None, True
for pp in PP:
    for T in TS:
        j = doc(os.path.join(PH, "kiem", "%s_T%s.json" % (pp, T)))
        if j:
            m_ = mt(j)
            if MT_REF is None:
                MT_REF = m_
            mt_ok = mt_ok and m_ == MT_REF and isinstance(m_["sha256"], dict) and len(m_["sha256"]) >= 11
SHA_MA = next(iter(sha_kiem)) if len(sha_kiem) == 1 else None
kiem_dat = all(kiem.values()) and la_sha(SHA_MA) and sha_txt.get("zcu104_doithu_ktnr.py") == SHA_MA and mt_ok
dung_luong = lambda j, L: (j.get("torch_luong") == int(L) and all((j.get("bien_moi_truong") or {}).get(k) == str(L)
                           for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) and (j.get("bien_moi_truong") or {}).get("CUDA_VISIBLE_DEVICES") == "")

# ---- so ghi ----
cfg = {}
try:
    for row in csv.reader(open(os.path.join(PH, "so_ghi.tsv")), delimiter="\t"):
        ph, pp, T, L, r, sd, n, rc, f = row
        cfg.setdefault((pp, T, L), []).append(dict(ph=ph, r=int(r), sd=int(sd), n=int(n), rc=int(rc), f=f))
except Exception as ex:
    sys.exit("so ghi hong: %s" % ex)


def n_du_kien(pp, T, L):
    if pp != "edgoc":
        return 200
    j = doc(os.path.join(PH, "thamdo_edgoc_T%s_L%s.json" % (T, L)))
    if not (j and j.get("phien") == ten_ph and j.get("pp") == "edgoc" and abs(j.get("T", -1) - float(T)) < 1e-9 and j.get("luong") == int(L)
            and j.get("n") == 3 and (j.get("tong") or {}).get("n") == 3 and j.get("khoi_dong") == 1 and j.get("luot_diem_hong") == 0
            and j.get("cuda") is False and j.get("script_sha256") == SHA_MA and mt(j) == MT_REF
            and dung_luong(j, L) and huu((j.get("tong") or {}).get("tb_ms"))):
        return None
    return 20 if j["tong"]["tb_ms"] > 1000 else 200


def hop_le(pp, T, L):
    v = sorted(cfg.get((pp, T, L), []), key=lambda x: x["r"])
    nd = n_du_kien(pp, T, L)
    if nd is None:
        return None, "thieu/hong tep tham do edgoc"
    if [x["r"] for x in v] != list(range(1, 8)) or any(x["ph"] != ten_ph or x["rc"] != 0 or x["sd"] != 20261107 + x["r"] or x["n"] != nd for x in v):
        return None, "so ghi thieu, ma thoat khac 0, seed/r/n khong dung"
    J = []
    for x in v:
        p = x["f"] if os.path.isabs(x["f"]) else os.path.join(goc_dir, x["f"])
        j = doc(p)
        if j is None:
            return None, "thieu hoac hong tep JSON"
        t = j.get("tong") if isinstance(j.get("tong"), dict) else {}
        ok = (mt(j) == MT_REF and dung_luong(j, L) and j.get("phan") == "doithu_dau_cuoi" and j.get("phien") == ten_ph and j.get("script_sha256") == SHA_MA
              and j.get("pp") == pp and abs(j.get("T", -1) - float(T)) < 1e-9 and j.get("luong") == int(L) and j.get("seed") == x["sd"]
              and j.get("n") == nd and t.get("n") == nd and j.get("khoi_dong") == 20 and j.get("luot_diem_hong") == 0 and j.get("cuda") is False
              and all(huu(t.get(k)) for k in ("tb_ms", "trung_vi_ms", "p95_ms", "max_ms")))
        if pp == "rcs":
            h = j.get("hoc_lai_lo20_tich_luy") if isinstance(j.get("hoc_lai_lo20_tich_luy"), dict) else {}
            ok = ok and h.get("n") == 12 and all(huu(h.get(k)) for k in ("tb_ms", "max_ms"))
        if not ok:
            return None, "JSON khong khop so ghi/phien/ban ma"
        J.append(j)
    return J, None


out = dict(phien=ten_ph, script_sha256=SHA_MA, kiem_tuong_duong=kiem, kiem_dat=kiem_dat, cau_hinh={}, so_sanh={})
for pp in PP:
    for T in TS:
        for L in LS:
            k = "%s_T%s_L%s" % (pp, T, L)
            if not kiem_dat:
                out["cau_hinh"][k] = dict(hop_le=False, ly_do="kiem tuong duong khong dat hoac khong cung ban ma"); continue
            try:
                J, ly_do = hop_le(pp, T, L)
            except Exception as ex:
                J, ly_do = None, "loi doc ho so: %s" % type(ex).__name__
            if J is None:
                out["cau_hinh"][k] = dict(hop_le=False, ly_do=ly_do); continue
            tb = [j["tong"]["tb_ms"] for j in J]
            e = dict(hop_le=True, n_luot=J[0]["n"], tb_ms=float(np.mean(tb)), sd_qua_tien_trinh=float(np.std(tb, ddof=1)), tb_min=float(min(tb)), tb_max=float(max(tb)),
                     p95_max_ms=float(max(j["tong"]["p95_ms"] for j in J)), max_ms=float(max(j["tong"]["max_ms"] for j in J)),
                     vuot_khoang_nghi=bool(max(j["tong"]["p95_ms"] for j in J) > 500), moi_truong=J[0]["moi_truong"])
            if pp == "rcs":
                e["hoc_lai_lo20_tich_luy_tb_ms"] = float(np.mean([j["hoc_lai_lo20_tich_luy"]["tb_ms"] for j in J]))
                e["hoc_lai_lo20_tich_luy_max_ms"] = float(max(j["hoc_lai_lo20_tich_luy"]["max_ms"] for j in J))
            out["cau_hinh"][k] = e
C_ = out["cau_hinh"]
for T in TS:
    for L in LS:
        m = C_["ktnr_T%s_L%s" % (T, L)]
        for pp in DOI:
            k = "%s_T%s_L%s" % (pp, T, L); x = C_[k]
            if not (m["hop_le"] and x["hop_le"]):
                out["so_sanh"][k] = None; continue
            out["so_sanh"][k] = dict(cssr_nhanh_hon=bool(m["tb_max"] < x["tb_min"]), ti_le_tb=x["tb_ms"] / m["tb_ms"],
                                     ti_le_phep_nhan=(MAC[pp] / MAC["ktnr"]) if T == "1.0" else None)
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in C_.items():
    print(k, ("TB %.2f ms, p95 max %.2f" % (v["tb_ms"], v["p95_max_ms"])) if v["hop_le"] else ("KHONG HOP LE: " + v["ly_do"]))
print(json.dumps(out["so_sanh"], ensure_ascii=False))
