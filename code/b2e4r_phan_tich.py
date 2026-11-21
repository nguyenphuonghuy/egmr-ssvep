#!/usr/bin/env python
"""Phan tich B2 E4R: E4R-b - KTNR, E4R-a - KTNR, E4R-b - E4R-a, E4 - E0, KTNR - KTN; P4 (DCR) moi nhanh.
Bao kem: 0,6 s; G2 = E4Rb - E4Ra; G3 = E4Ra - KTNR .
CI: BCa 10.000 lan, seed 20261107, qua nguoi (don vi diem %). BCa suy bien: "khong tinh duoc", khong xep nhanh nao.
Dung: python b2e4r_phan_tich.py <json ra> <tep ket qua>... (moi tep mot bo, mot cua so)
"""
import json, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107


def ci(a):
    a = 100 * np.asarray(a, float)
    tb = float(a.mean())
    if not np.isfinite(a).all() or np.allclose(a, a[0]):
        return dict(tb=tb, thap=None, cao=None, ghi_chu="khong tinh duoc")
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    except Exception as e:
        return dict(tb=tb, thap=None, cao=None, ghi_chu="khong tinh duoc: " + str(e)[:80])
    if not (np.isfinite(r.low) and np.isfinite(r.high)):
        return dict(tb=tb, thap=None, cao=None, ghi_chu="khong tinh duoc")
    return dict(tb=tb, thap=float(r.low), cao=float(r.high))


SO_NGUOI = {"benchmark": 35, "beta": 70, "dong": 59}


def main():
    import os, hashlib
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    ra = os.path.abspath(sys.argv[1]); vao = [os.path.abspath(x) for x in sys.argv[2:]]
    assert ra not in vao and not os.path.exists(ra), "tep ra trung dau vao hoac da co: khong ghi de"
    out = dict(tep_vao={}, o={})
    for f in sys.argv[2:]:
        j = json.load(open(f))
        assert j["alpha"] == 0.5 and j["H"] == 5 and j["k"] == 5 and j["n0"] == 40 and j.get("du_fold") is True, f"{f}: cau hinh khac hoac chay thu"
        p = j["tung_nguoi"]
        key = f"{j['ds']}_{j['T']}"
        assert key not in out["o"], f"o {key} trung: tu choi"
        assert [x["fold"] for x in p] == list(range(1, SO_NGUOI[j["ds"]] + 1)), f"{f}: thieu, trung hoac sai thu tu fold"
        assert all(x["seed"] == 20261107 + 1000 * x["fold"] + int(round(j["T"] * 10)) for x in p), f"{f}: seed lech"
        assert all(np.isfinite([x[k] for k in ("KTN", "KTNR", "E0", "E4", "E4Ra", "E4Rb", "P4_a", "P4_b")]).all() for x in p), f"{f}: chi so khong huu han"
        out["tep_vao"][key] = dict(tep=f, sha256=sha(f), so_luot_ktnr_khong_chung_nhan=int(sum(x["so_luot_ktnr_khong_chung_nhan"] for x in p)),
                                   so_luot_ktn_khong_chung_nhan=int(sum(x.get("so_luot_ktn_khong_chung_nhan", 0) for x in p)))
        if j["ds"] == "dong":                                                                             # bat bien so dem cho C0
            assert all(isinstance(x["tinh_dung"], int) and x["n_cham"] == 120 and 0 <= x["tinh_dung"] <= 120 for x in p), f"{f}: so dem C0 khong hop le"
        v = lambda k: np.array([x[k] for x in p])
        o = dict(n=len(p), TB={k: 100 * float(v(k).mean()) for k in ("KTN", "KTNR", "E0", "E4", "E4Ra", "E4Rb")},
                 E4Rb_tru_KTNR=ci(v("E4Rb") - v("KTNR")), E4Ra_tru_KTNR=ci(v("E4Ra") - v("KTNR")), G2_E4Rb_tru_E4Ra=ci(v("E4Rb") - v("E4Ra")),
                 E4_tru_E0=ci(v("E4") - v("E0")), KTNR_tru_KTN=ci(v("KTNR") - v("KTN")),
                 P4_b=float(v("P4_b").mean()), P4_a=float(v("P4_a").mean()),
                 ti_le_nguoi_b_hon=float((v("E4Rb") > v("KTNR")).mean()), ti_le_nguoi_b_kem=float((v("E4Rb") < v("KTNR")).mean()))
        if j["ds"] == "dong":
            o["tinh_dung"] = int(sum(x["tinh_dung"] for x in p)); o["tinh_mau"] = int(sum(x["n_cham"] for x in p))
        out["o"][key] = o
        print(f"{key}: n={o['n']} KTNR {o['TB']['KTNR']:.2f} E4Ra {o['TB']['E4Ra']:.2f} E4Rb {o['TB']['E4Rb']:.2f} | "
              f"b-KTNR {o['E4Rb_tru_KTNR']} P4 {o['P4_b']:.3f} | a-KTNR {o['E4Ra_tru_KTNR']} P4 {o['P4_a']:.3f}")
    with open(ra, "x", encoding="utf-8") as fo:                                                           # "x": that bai neu tep da co
        json.dump(out, fo, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
