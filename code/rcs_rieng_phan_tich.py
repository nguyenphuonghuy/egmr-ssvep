#!/usr/bin/env python
"""Phan tich BO SUNG cho H3-b voi cach thu gon thu ba (TRCA khong gop lop).
Chenh ghep cap theo nguoi (S1..Sn), luot 41..het, CI 95 % BCa 10.000 lan, seed 20261107.
Dung: python rcs_rieng_phan_tich.py <thu muc rcs> <thu muc ket-qua-b1moi> <json ra>
"""
import json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RCS, CU, RA = sys.argv[1:4]
BT = ["tranK_lo", "tranKR_lo", "tranK_luot", "tranKR_luot"]


def ci(a):
    a = 100 * np.asarray(a, float)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high))


out = dict(seed=SEED, bo={})
chinh = []
for ds in ("benchmark", "beta"):
    r = json.load(open(os.path.join(RCS, ds + "_rcs-rieng_seed20261107_20261114.json"), encoding="utf-8"))
    k = json.load(open(os.path.join(CU, ds + "_b1moi_buoc0b_seed20261107_20261114.json"), encoding="utf-8"))
    assert r["n_subjects"] == k["n_subjects"] and r["n0"] == 40 and r["nguoi"] == list(range(1, r["n_subjects"] + 1))
    n = {"benchmark": 240, "beta": 160}[ds]
    out["bo"][ds] = {}
    for T in ("0.6", "0.8", "1.0"):
        pr = r["ket_qua"][T]["tung_nguoi"]; pk = [p["40"] for p in k["ket_qua"][T]["tung_nguoi"]]
        kh = np.array([p["kh_deu"] for p in pk])
        kh_ph = (40 * np.array([p["nhan_gia_dung_b2"] for p in pk]) + (n - 40) * kh) / n
        m = r["ket_qua"][T]["mac"]
        e = dict(tb={"kh_deu": float(100 * kh.mean())}, mac={"tranK": m["tranK_quyet_dinh"], "tranKR": m["tranKR_quyet_dinh"], "tranKR_so_hai": m["tranKR_so_hai"]})
        for v in BT:
            a = np.array([p[v]["sau"] for p in pr]); b = np.array([p[v]["phien"] for p in pr])
            e["tb"][v] = float(100 * a.mean()); e["tb"][v + "_phien"] = float(100 * b.mean())
            e["kh_deu_tru_" + v] = ci(kh - a); e["phien_kh_deu_tru_" + v] = ci(kh_ph - b)
            if T == "1.0" and v in ("tranK_lo", "tranKR_lo"):
                chinh.append(e["kh_deu_tru_" + v])
        out["bo"][ds][T] = e
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
for ds, d in out["bo"].items():
    for T, e in d.items():
        print(ds, T, {k: round(v, 1) for k, v in e["tb"].items()}, e["mac"])
        print("    ", " | ".join("%s %s" % (v, f(e["kh_deu_tru_" + v])) for v in BT))
        print("     ca phien:", " | ".join("%s %s" % (v, f(e["phien_kh_deu_tru_" + v])) for v in BT))
