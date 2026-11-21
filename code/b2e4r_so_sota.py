#!/usr/bin/env python
"""So EGM-R (E4Rb trong tep ket qua) voi SOTA khong hieu chuan tai hien cung giao thuc. PHEP SO THAM DO (quy tac dien dat ghi trong
moi bo, moi cua so neu rieng; can duoi > 0 'hon', can tren < 0 'kem',
con lai 'chua phan biet'. Cham luot 41 den het, chenh ghep cap theo nguoi, BCa 10.000, seed 20261107 (cung ham ci voi b1_ci_bo_sung.py).
Dung: python b2e4r_so_sota.py <json ra> <thu muc ket-qua-b2e4r> <thu muc ket-qua-b1moi>
Nguon theo nguoi (cung cach nap voi ma B1 da dung):
  EGM-R: <b2>/<bo>_b2e4r_<T>_*.json (Dong: <b2>/dong/), 'tung_nguoi'[i]['E4Rb'], fold 1..n.
  Kiem ghep nguoi: 'KTNR' cua tep B2 phai trung 'ktnr' cua ktnr/<bo>_ktnr-buoc1_<T>_*.json (Dong: ktnr/dong_ktnr-xn_<T>_*.json).
  RCS day du: rcs/<bo>_rcs-tai-hien_*.json ['ket_qua'][T]['tung_nguoi'][i]['day_du_lo']['sau'].
  FBCCA: fbcca-dung_seed20261107_20261118.json ['bo'][bo][T]['tung_nguoi'][i]['sau'] (cong thuc dung rho^2; KHONG dung 'fbcca5_sau'
  cua tep RCS vi cong rho chua binh phuong).
  Ensemble-DNN goc: ed/du_<bo>_<T>/fold*.json ['sau'] (Benchmark, BETA).
  CCA-Net: ccanet/gt_<bo>_<T>/fold*.json ['final']['sau'], config che == 'gt'.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RA, B2, B1 = sys.argv[1:4]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
N = {"benchmark": 35, "beta": 70, "dong": 59}
RCS = {"benchmark": "benchmark_rcs-tai-hien_seed20261107_20261114.json", "beta": "beta_rcs-tai-hien_seed20261107_20261114.json",
       "dong": "dong_rcs-tai-hien_seed20261107_20261117.json"}


def ci(a):
    a = 100 * np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()), phuong_phap="hang")
    pp = "BCa"
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError("khong huu han")
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


def xep(c):
    return "hon" if c["thap"] > 0 else ("kem" if c["cao"] < 0 else "chua phan biet")


def hop_le(v, n, ten):
    v = np.asarray(v, float)
    assert len(v) == n and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), ten
    return v


def mot(pat):
    fs = [f for f in glob.glob(pat) if "phan-tich" not in f]
    assert len(fs) == 1, (pat, fs)
    return fs[0]


def thu_muc_fold(d, n, ds, T, che=None):
    if che:
        cfg = json.load(open(os.path.join(d, "config.json")))
        assert cfg["ds"] == ds and abs(cfg["T"] - T) < 1e-9 and cfg["che"] == che and not cfg["du_nhanh"], d
    J = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(d, "fold*.json")))]
    assert [j["fold"] for j in J] == list(range(1, n + 1)), (d, len(J))
    for j in J:
        assert j["ds"] == ds and abs(j["T"] - T) < 1e-9, (d, j["fold"])
    return J


out = dict(seed=SEED, ghi_chu="phep so tham do", tep={}, o={})
for ds in ("benchmark", "beta", "dong"):
    n = N[ds]
    R = json.load(open(os.path.join(B1, "rcs", RCS[ds]), encoding="utf-8")); out["tep"][RCS[ds]] = True
    FB = json.load(open(os.path.join(B1, "fbcca-dung_seed20261107_20261118.json"), encoding="utf-8"))["bo"][ds]
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        f2 = mot(os.path.join(B2, "dong" if ds == "dong" else "", "%s_b2e4r_%s_seed20261107_*.json" % (ds, tt)))
        P2 = json.load(open(f2, encoding="utf-8"))
        assert P2["ds"] == ds and abs(P2["T"] - float(T)) < 1e-9
        P2 = P2["tung_nguoi"]; assert [p["fold"] for p in P2] == list(range(1, n + 1))
        egm = hop_le([p["E4Rb"] for p in P2], n, "E4Rb")
        f1 = mot(os.path.join(B1, "ktnr", ("dong_ktnr-xn_%s_seed20261107_*.json" if ds == "dong" else ds + "_ktnr-buoc1_%s_seed20261107_*.json") % tt))
        P1 = json.load(open(f1, encoding="utf-8"))["tung_nguoi"]; assert [p["fold"] for p in P1] == list(range(1, n + 1))
        assert np.allclose([p["KTNR"] for p in P2], [p["ktnr"] for p in P1], atol=1e-12), ("ghep nguoi lech", ds, T)
        out["tep"][os.path.basename(f2)] = out["tep"][os.path.basename(f1)] = True
        dt = {}
        tr = R["ket_qua"][T]["tung_nguoi"]; assert len(tr) == n
        dt["RCS"] = hop_le([q["day_du_lo"]["sau"] for q in tr], n, "rcs")
        fq = FB[T]["tung_nguoi"]; assert [q["nguoi"] for q in fq] == list(range(1, n + 1))
        dt["FBCCA"] = hop_le([q["sau"] for q in fq], n, "fbcca")
        if ds != "dong":
            dt["EnsDNN"] = hop_le([j["sau"] for j in thu_muc_fold(os.path.join(B1, "ed", "du_%s_%s" % (ds, tt)), n, ds, float(T))], n, "ed")
        dt["CCANet"] = hop_le([j["final"]["sau"] for j in thu_muc_fold(os.path.join(B1, "ccanet", "gt_%s_%s" % (ds, tt)), n, ds, float(T), "gt")], n, "ccanet")
        e = dict(tb=dict(EGM_R=100 * float(egm.mean()), **{k: 100 * float(v.mean()) for k, v in dt.items()}), so={})
        for k, v in dt.items():
            c = ci(egm - v); c["xep"] = xep(c); e["so"][k] = c
        e["mac"] = dict(RCS=R["ket_qua"][T]["mac"].get("day_du_quyet_dinh"), FBCCA=FB[T].get("mac_5dai"))
        out["o"]["%s_%s" % (ds, T)] = e
        print(ds, T, {k: round(v, 2) for k, v in e["tb"].items()})
        print("    ", " | ".join("%s %+.2f [%+.2f; %+.2f] %s" % (k, c["tb"], c["thap"], c["cao"], c["xep"]) for k, c in e["so"].items()), e["mac"])
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
