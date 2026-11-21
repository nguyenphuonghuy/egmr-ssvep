"""Phan tich I, J (bo sung 2026-11-21) va tom tat K (ZCU104). CI 95 % BCa qua nguoi, 10 000 lan, seed 20261107."""
import collections, glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap
sys.stdout.reconfigure(encoding="utf-8")
SEED = 20261107
K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ket-qua-b2e4r", "bo-sung-20261121")


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
        p = glob.glob(os.path.join(K, f"{ds}_b2e4r-huong_T{T}_*.json"))
        if not p:
            print(ds, T, "chua co"); continue
        t = json.load(open(p[0]))["tung_nguoi"]
        g = lambda k: np.array([r[k] for r in t]); h = lambda k: np.array([r[k]["acc"] for r in t])
        u, e, i1, i2, j = g("KTNR"), g("E4Rb"), h("I1_ngau_nhien"), h("I2_nguoc_dau"), h("J_l2_chuan_hoa")
        o = {"EGM-R - UM-R": ci(e - u), "I1 - UM-R": ci(i1 - u), "EGM-R - I1": ci(e - i1), "I2 - UM-R": ci(i2 - u), "EGM-R - I2": ci(e - i2),
             "J - UM-R": ci(j - u), "EGM-R - J": ci(e - j)}
        ra[f"{ds}_{T}"] = dict(so=o, n=len(t), tv_I2=float(np.mean([r["I2_nguoc_dau"]["tv"] for r in t])))
        print(f"== {ds} T{T} n={len(t)}")
        for k, c in o.items():
            print("   ", k, f(c))
p = os.path.join(K, "..", "zcu104", "phien_mot-lan-l2_chinh_thuc_1", "mot_lan_l2.jsonl")
if os.path.exists(p):
    d = collections.defaultdict(list)
    for L in open(p):
        j = json.loads(L); d[(j["che_do"], j["Tphien"])].append(j)
    for k, v in sorted(d.items()):
        for q in v[0]["ket_qua"]:
            print("ZCU104", k, q, "max p95 %.1f ms" % max(x["ket_qua"][q]["p95_ms"] for x in v), "n", len(v), "byte", v[0]["byte"])
        ra[f"K_{k[0]}_{k[1]}"] = {q: max(x["ket_qua"][q]["p95_ms"] for x in v) for q in v[0]["ket_qua"]}
json.dump(ra, open(os.path.join(K, "phan-tich_huong_20261121.json"), "w"), indent=1)
