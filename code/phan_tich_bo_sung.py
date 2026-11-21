"""Phan tich A, C, D bo sung 2026-11-20. CI 95 % BCa qua nguoi, 10 000 lan, seed 20261107 (nhu b2e4r_phan_tich.py).
Doc JSON da tai ve results/ket-qua-b2e4r/bo-sung-20261120/, ghi bang ra results/ket-qua-b2e4r/bo-sung-20261120/phan-tich_bo-sung_20261120.json va in tom tat."""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap
sys.stdout.reconfigure(encoding="utf-8")
SEED = 20261107
K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ket-qua-b2e4r", "bo-sung-20261120")
GOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ket-qua-b2e4r") + "/"


def ci(a):
    a = 100 * np.asarray(a, float); tb = float(a.mean())
    if not np.isfinite(a).all() or np.allclose(a, a[0]):
        return dict(tb=tb, thap=None, cao=None)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=tb, thap=float(r.low), cao=float(r.high))


f = lambda c: "%+.2f [%+.2f, %+.2f]" % (c["tb"], c["thap"], c["cao"]) if c["thap"] is not None else "%+.2f [khong tinh duoc]" % c["tb"]
ra = {}
print("== A: doi chung he so")
for ds in ("benchmark", "beta", "dong"):
    for T in ("10", "06"):
        p = glob.glob(os.path.join(K, f"{ds}_b2e4r-dclam_T{T}_*.json"))
        if not p:
            print(ds, T, "chua co"); continue
        t = json.load(open(p[0]))["tung_nguoi"]
        g = lambda k: np.array([r[k] for r in t])
        o = {"EGM-R - UM-R": ci(g("E4Rb") - g("KTNR")), "A1 - UM-R": ci(g("A1") - g("KTNR")), "EGM-R - A1": ci(g("E4Rb") - g("A1")),
             "A2 - UM-R": ci(g("A2") - g("KTNR")), "EGM-R - A2": ci(g("E4Rb") - g("A2"))}
        tv = {k: float(np.mean(g(k))) for k in ("tv_A1", "tv_A2", "tv_b")}
        ra[f"A_{ds}_{T}"] = dict(so=o, tv=tv, n=len(t), dich_g=int(sum(r["dich_g"] for r in t)),
                                 do_chinh_xac={k: float(100 * g(k).mean()) for k in ("KTNR", "E4Rb", "A1", "A2")})
        print(ds, T, "n", len(t), "| TV A1 %.3f A2 %.3f EGM-R %.3f" % (tv["tv_A1"], tv["tv_A2"], tv["tv_b"]), "| dich g:", ra[f"A_{ds}_{T}"]["dich_g"])
        for k, c in o.items():
            print("   ", k, f(c))
print("== C: OACCA-NQ tren Dong")
p = glob.glob(os.path.join(K, "dong_oacca-nq_*.json"))
if p:
    oa = json.load(open(p[0]))["tung_nguoi"]
    for T, TT in (("10", "1.0"), ("06", "0.6")):
        e = json.load(open(GOC + f"dong/dong_b2e4r_T{T}_seed20261107_20261117.json"))["tung_nguoi"]
        a = np.array([r["T"][TT]["oacca"]["sau"] for r in oa]); b = np.array([r["E4Rb"] for r in e])
        assert [r["nguoi"] for r in oa] == [r["fold"] for r in e]
        ra[f"C_{T}"] = dict(oacca=float(100 * a.mean()), egmr=float(100 * b.mean()), hieu=ci(b - a))
        print(TT, "OACCA-NQ %.2f | EGM-R %.2f | EGM-R - OACCA-NQ" % (100 * a.mean(), 100 * b.mean()), f(ci(b - a)))
print("== D: hoan vi tren Dong")
for T in ("10", "06"):
    p = glob.glob(os.path.join(K, f"dong_b2e4r-hoanvi_T{T}_*.json"))
    if not p:
        print(T, "chua co"); continue
    t = json.load(open(p[0]))["tung_nguoi"]
    for P in (1, 2):
        d, q = [], []
        for r in t:
            h = r["hoan_vi"]; d0 = [x for x in h if x["r"] == 0][0]; D0 = d0["E4Rb"] - d0["KTNR"]
            dr = np.mean([x["E4Rb"] - x["KTNR"] for x in h if x["P"] == P and x["r"] > 0])
            d.append(dr); q.append(dr - D0)
        ra[f"D_{T}_P{P}"] = dict(D=ci(d), Q=ci(q))
        print(T, "P%d" % P, "D", f(ci(d)), "| Q", f(ci(q)))
json.dump(ra, open(os.path.join(K, "phan-tich_bo-sung_20261120.json"), "w"), indent=1)
