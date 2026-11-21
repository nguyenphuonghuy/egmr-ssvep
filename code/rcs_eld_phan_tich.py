#!/usr/bin/env python
"""Phan tich H3a, H3b tren eldBETA (bo CHUA xem voi RCS).
100 nguoi, luot 41..63, chenh ghep cap theo nguoi, CI 95 % BCa 10.000 lan, seed 20261107, o 1,0 s (0,6 va 0,8 s bao kem).
Dung: python rcs_eld_phan_tich.py <json KTN T10> <json KTN T06> <thu muc rcs eld> <json ra>
"""
import json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
K10, K06, D, RA = sys.argv[1:5]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")


def ci(a):
    a = 100 * np.asarray(a, float)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high))


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
r1 = json.load(open(os.path.join(D, "eldbeta_rcs-tai-hien_seed20261107_20261115.json"), encoding="utf-8"))
r2 = json.load(open(os.path.join(D, "eldbeta_rcs-rieng_seed20261107_20261115.json"), encoding="utf-8"))
for r_ in (r1, r2):                                                          # buoc dung giao thuc, du 100 hang
    assert r_["ds"] == "eld" and r_["n_subjects"] == 100 and r_["nguoi"] == list(range(1, 101)) and r_["n0"] == 40 and r_["seed"] == SEED
    assert all(len(r_["ket_qua"][T_]["tung_nguoi"]) == 100 for T_ in ("0.6", "1.0"))


def vec(x):
    x = np.asarray(x, float)
    assert x.shape == (100,) and np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
    return x


out = dict(seed=SEED, T={})
for T, fn in (("1.0", K10), ("0.6", K06)):
    j = json.load(open(fn, encoding="utf-8")); per = j["tung_nguoi"]
    assert j["ds"] == "eld" and len(per) == 100 and [p["fold"] for p in per] == list(range(1, 101)) and abs(j["T"] - float(T)) < 1e-9
    assert j["xac_nhan"] and j["n0"] == 40 and j["seed"] == SEED and j["k_src"] == 5 and j["buoc_nguon"] == 100
    ktn = vec([p["B"]["SF"]["sau"] for p in per])
    rc = {v: vec([p[v]["sau"] for p in r1["ket_qua"][T]["tung_nguoi"]]) for v in ("tran_lo", "tran1R_lo", "tranG_lo", "day_du_lo") if v in r1["ket_qua"][T]["tung_nguoi"][0]}
    rc.update({v: vec([p[v]["sau"] for p in r2["ket_qua"][T]["tung_nguoi"]]) for v in ("tranK_lo", "tranKR_lo")})
    assert "day_du_lo" in rc
    m1, m2 = r1["ket_qua"][T]["mac"], r2["ket_qua"][T]["mac"]
    mac = dict(tran_lo=m1["tran_quyet_dinh"], tran1R_lo=m1.get("tran1R_quyet_dinh"), tranG_lo=m1.get("tranG_quyet_dinh"), tranK_lo=m2["tranK_quyet_dinh"], tranKR_lo=m2["tranKR_quyet_dinh"], day_du_lo=m1["day_du_quyet_dinh"])
    trong_tran = [v for v in ("tran_lo", "tran1R_lo", "tranG_lo", "tranK_lo", "tranKR_lo") if v in rc and mac[v] is not None and mac[v] <= 1000000]
    manh = max(trong_tran, key=lambda v: rc[v].mean())
    e = dict(tb=dict(KTN=float(100 * ktn.mean()), **{v: float(100 * x.mean()) for v, x in rc.items()}), mac=mac, rcs_trong_tran_manh_nhat=manh,
             H3a=ci(ktn - rc[manh]), H3b=ci(ktn - rc["day_du_lo"]), rep_wR=float(100 * np.mean([p["rep_wR"] for p in r1["ket_qua"][T]["tung_nguoi"]])))
    out["T"][T] = e
    print(T, {a: round(b, 1) for a, b in e["tb"].items()}, "manh nhat:", manh, mac)
    print("    H3a KTN-%s" % manh, f(e["H3a"]), "| H3b KTN-RCS day du", f(e["H3b"]))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
