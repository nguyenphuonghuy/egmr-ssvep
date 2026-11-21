"""Phan tich F, G, H (bo sung 2026-11-20). CI 95 % BCa qua nguoi, 10 000 lan, seed 20261107."""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap
sys.stdout.reconfigure(encoding="utf-8")
SEED = 20261107
K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ket-qua-b2e4r", "bo-sung-20261120")


def ci(a):
    a = 100 * np.asarray(a, float); tb = float(a.mean())
    if not np.isfinite(a).all() or np.allclose(a, a[0]):
        return dict(tb=tb, thap=None, cao=None)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=tb, thap=float(r.low), cao=float(r.high))


f = lambda c: "%+.2f [%+.2f, %+.2f]" % (c["tb"], c["thap"], c["cao"]) if c["thap"] is not None else "%+.2f [khong tinh duoc]" % c["tb"]
ra = {}
for ds in ("benchmark", "beta", "dong"):
    for T in ("10", "06"):
        p = glob.glob(os.path.join(K, f"{ds}_b2e4r-l2-T_T{T}_*.json"))
        if not p:
            print(ds, T, "chua co"); continue
        t = json.load(open(p[0]))["tung_nguoi"]
        u = np.array([r["KTNR"] for r in t]); e = np.array([r["E4Rb"] for r in t])
        o = {"EGM-R - UM-R": ci(e - u)}
        for eta in t[0]["l2"]:
            l = np.array([r["l2"][eta]["acc"] for r in t])
            o[f"l2 eta={eta} - UM-R"] = ci(l - u); o[f"EGM-R - l2 eta={eta}"] = ci(e - l)
        for k in t[0]["T_thay"]:
            g = np.array([r["T_thay"][k]["acc"] for r in t])
            o[f"EGM-R({k}) - UM-R"] = ci(g - u)
        a40 = float(100 * np.mean([r["tinh_40_dau"] for r in t]))
        ra[f"{ds}_{T}"] = dict(so=o, tinh_40_dau=a40, n=len(t))
        print(f"== {ds} T{T} n={len(t)} | mang chung 40 luot dau {a40:.2f} %")
        for k, c in o.items():
            print("   ", k, f(c))
json.dump(ra, open(os.path.join(K, "phan-tich_l2-T_20261120.json"), "w"), indent=1)
