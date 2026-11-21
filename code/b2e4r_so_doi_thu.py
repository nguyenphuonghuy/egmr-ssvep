#!/usr/bin/env python
"""So E4R-b voi OACCA-NQ (H5R) va SFDA-TT (H6R).
Khong chay moi; doc so tung nguoi da co. Chenh ghep cap theo nguoi, CI 95 % BCa 10.000 lan, seed 20261107.
Dung: python b2e4r_so_doi_thu.py <thu muc ket-qua-b2e4r> <thu muc ket-qua-b1gop/d6> <json ra>
"""
import hashlib, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
N = {"benchmark": 35, "beta": 70}
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
doc = lambda f: json.load(open(f, encoding="utf-8"))


def ci(a):
    a = 100 * np.asarray(a, float)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    assert np.isfinite(r.low) and np.isfinite(r.high)
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high))


def xep(c):
    return "hon" if c["thap"] > 0 else ("kem" if c["cao"] <= 0 else "chua phan biet")


def main():
    E, D6, ra = sys.argv[1:4]
    assert not os.path.exists(ra), "tep ra da co: khong ghi de"
    out = dict(tep={}, o={}, cau={})
    for ds in ("benchmark", "beta"):
        n = N[ds]
        fo = os.path.join(D6, f"oacca_nq_{ds}.json"); o = doc(fo); out["tep"][fo] = sha(fo)
        assert o["ds"] == ds and o["che_do"] == "nq" and o["n0"] == 40 and [p["nguoi"] for p in o["tung_nguoi"]] == list(range(1, n + 1))
        for T, tag in (("1.0", "T10"), ("0.6", "T06")):
            fe = os.path.join(E, f"{ds}_b2e4r_{tag}_seed20261107_20261117.json"); e = doc(fe); out["tep"][fe] = sha(fe)
            ft = os.path.join(D6, f"sfda_tt_{ds}_{tag}.json"); t = doc(ft); out["tep"][ft] = sha(ft)
            assert e["ds"] == ds and abs(e["T"] - float(T)) < 1e-9 and e["n0"] == 40 and [p["fold"] for p in e["tung_nguoi"]] == list(range(1, n + 1))
            assert t["ds"] == ds and t["che_do"] == "tt" and abs(t["T"] - float(T)) < 1e-9 and [p["fold"] for p in t["tung_nguoi"]] == list(range(1, n + 1))
            assert t["trongso"] == e["trongso"], "SFDA-TT phai dung cung thu muc mang nen voi E4R-b"
            assert [p["trongso_sha256"] for p in t["tung_nguoi"]] == e["trongso_fold_sha256"], "ma bam trong so tung fold lech"
            eb = np.array([p["E4Rb"] for p in e["tung_nguoi"]])
            oa = np.array([p["T"][T]["oacca"]["sau"] for p in o["tung_nguoi"]])
            sd = np.array([p["sfda"]["sau"] for p in t["tung_nguoi"]])
            for v in (eb, oa, sd):
                assert v.shape == (n,) and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
            r = dict(tb=dict(E4Rb=float(100 * eb.mean()), OACCA_NQ=float(100 * oa.mean()), SFDA_TT=float(100 * sd.mean())),
                     H5R=ci(eb - oa), H6R=ci(eb - sd))
            r["H5R"]["xep"] = xep(r["H5R"]); r["H6R"]["xep"] = xep(r["H6R"])
            r["ti_le_nguoi_hon"] = dict(OACCA_NQ=float((eb > oa).mean()), SFDA_TT=float((eb > sd).mean()))
            out["o"][f"{ds}_{T}"] = r
            f = lambda c: f"{c['tb']:+.2f} [{c['thap']:+.2f}; {c['cao']:+.2f}] {c['xep']}"
            print(f"{ds} {T}: E4R-b {r['tb']['E4Rb']:.2f} OACCA-NQ {r['tb']['OACCA_NQ']:.2f} SFDA-TT {r['tb']['SFDA_TT']:.2f} | H5R {f(r['H5R'])} | H6R {f(r['H6R'])}")
    for h, ten in (("H5R", "OACCA-NQ"), ("H6R", "SFDA-TT")):
        x = {ds: out["o"][f"{ds}_1.0"][h]["xep"] for ds in ("benchmark", "beta")}
        if all(v == "hon" for v in x.values()):
            out["cau"][h] = f"E4R-b hon {ten} tren Benchmark va BETA (1,0 s)"
        else:
            out["cau"][h] = f"neu rieng tung bo (1,0 s): " + "; ".join(f"{ds} {v}" for ds, v in x.items()) + "; khong gop hai bo"
    with open(ra, "x", encoding="utf-8") as fo_:
        json.dump(out, fo_, indent=1, ensure_ascii=False)
    print("CAU:", out["cau"])


if __name__ == "__main__":
    main()
