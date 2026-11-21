"""Nhom 2: phan bo EGM-R - UM-R theo nguoi va ITR (Wolpaw) cua UM-R, EGM-R tren luot 41..het.
ITR moi nguoi: B = log2 N + P log2 P + (1 - P) log2((1 - P)/(N - 1)) bit/luot, chia cho (cua so + 0,5 s chuyen nhin), doi ra bit/phut.
Gia dinh 0,5 s chuyen nhin cho ca ba bo (bang khoang nghi giua hai luot trong bai). CI BCa nhu b2e4r_phan_tich.py."""
import json, os, sys
import numpy as np
from scipy.stats import bootstrap
sys.stdout.reconfigure(encoding="utf-8")
SEED, N, GS = 20261107, 40, 0.5
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ket-qua-b2e4r") + "/"
TEP = {"benchmark": G + "benchmark_b2e4r_T{T}_seed20261107_20261117.json", "beta": G + "beta_b2e4r_T{T}_seed20261107_20261117.json",
       "dong": G + "dong/dong_b2e4r_T{T}_seed20261107_20261117.json"}


def ci(a):
    a = np.asarray(a, float)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    return [float(a.mean()), float(r.low), float(r.high)]


def itr(p, t):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    b = np.log2(N) + p * np.log2(p) + np.where(p < 1, (1 - p) * np.log2(np.clip((1 - p) / (N - 1), 1e-300, None)), 0.0)
    b = np.where(p <= 1.0 / N, 0.0, b)                    # quy uoc thuong dung: duoi muc ngau nhien thi 0
    return b * 60.0 / (t + GS)


ra = {}
for ds, p in TEP.items():
    for T, t in (("10", 1.0), ("06", 0.6)):
        r = json.load(open(p.format(T=T), encoding="utf-8"))["tung_nguoi"]
        u = np.array([x["KTNR"] for x in r]); e = np.array([x["E4Rb"] for x in r]); n = np.array([x["n_cham"] for x in r])
        d = 100 * (e - u); dl = np.rint((e - u) * n).astype(int)
        iu, ie = itr(u, t), itr(e, t)
        ra[f"{ds}_{T}"] = dict(n=len(r), trung_vi=float(np.median(d)), p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)),
                               nho_nhat=float(d.min()), lon_nhat=float(d.max()), ti_le_kem=float((d < 0).mean()), ti_le_hon=float((d > 0).mean()),
                               luot_kem_nhat=int(dl.min()), luot_hon_nhat=int(dl.max()),
                               itr_umr=float(iu.mean()), itr_egmr=float(ie.mean()), itr_hieu=ci(ie - iu), d=d.round(3).tolist())
        o = ra[f"{ds}_{T}"]
        print(f"{ds} {t}s n={o['n']} trung vi {o['trung_vi']:+.2f} P10 {o['p10']:+.2f} min {o['nho_nhat']:+.2f} ({o['luot_kem_nhat']} luot) "
              f"max {o['lon_nhat']:+.2f} kem {100*o['ti_le_kem']:.0f}% hon {100*o['ti_le_hon']:.0f}% | ITR UM-R {o['itr_umr']:.1f} EGM-R {o['itr_egmr']:.1f} "
              f"hieu {o['itr_hieu'][0]:+.2f} [{o['itr_hieu'][1]:+.2f}, {o['itr_hieu'][2]:+.2f}] bit/phut")
K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ket-qua-b2e4r", "bo-sung-20261120")
os.makedirs(K, exist_ok=True)
json.dump(ra, open(os.path.join(K, "phan-bo-itr_20261120.json"), "w"), indent=1)
