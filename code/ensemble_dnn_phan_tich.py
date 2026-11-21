#!/usr/bin/env python
"""Cong tai hien va phan tich E1 cho Ensemble-DNN.
  cong <thu muc ED-du Benchmark T10>     : TB 'phien' (moi luot) qua 35 nguoi trong [0,72; 0,86] -> ma thoat 0, nguoc lai 5.
  e1 <json ra> <thu muc ket-qua-b1moi> <thu muc results/b1moi_ed (ban sao)> :
      Bao kem: Benchmark; 0,6 s; KTN - ED-du; ED-du - ED-22; cong tai hien.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
SO = {"benchmark": 35, "beta": 70}


def nap_dir(d, ds, bt, T):
    fs = sorted(glob.glob(os.path.join(d, "fold*.json")))
    J = [json.load(open(f)) for f in fs]
    assert len(J) == SO[ds] and [j["fold"] for j in J] == list(range(1, SO[ds] + 1)), (d, len(J))
    assert all(j["ds"] == ds and j["bien_the"] == bt and abs(j["T"] - T) < 1e-9 and j["n0"] == 40 and j["vong2"] == 1000 for j in J), d
    assert all(j["nch"] == (120 if bt == "du" else 22) and j["seed"] == SEED + 1000 * j["fold"] + int(round(T * 10)) for j in J), d
    assert len({j["script_sha256"] for j in J}) == 1 and len({json.dumps(j["ma_phu_sha256"], sort_keys=True) for j in J}) == 1, ("tron phien ban", d)
    if bt == "22":
        assert all(j["trongso_sha256"] for j in J), d
    for k_ in ("sau", "phien"):
        v_ = np.array([j[k_] for j in J]); assert np.isfinite(v_).all() and (v_ >= 0).all() and (v_ <= 1).all(), d
    if bt == "du":
        assert all(j["vong1"] == {"benchmark": 1000, "beta": 800}[ds] for j in J), d
    return J


def ci(a):
    a = 100 * np.asarray(a, float)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    lo_, hi_ = float(r.low), float(r.high)
    return dict(tb=float(a.mean()), thap=lo_, cao=hi_, huu_han=bool(np.isfinite(lo_) and np.isfinite(hi_)))


f = lambda c: "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
def cong(dd):
    J = nap_dir(dd, "benchmark", "du", 1.0)
    tb = float(np.mean([j["phien"] for j in J]))
    return tb, bool(0.72 <= tb <= 0.86)


if sys.argv[1] == "cong":
    tb, dat = cong(sys.argv[2])
    print("CONG TAI HIEN ED-du Benchmark 1,0 s, moi luot: %.4f" % tb, "DAT" if dat else "KHONG DAT", flush=True)
    sys.exit(0 if dat else 5)

RA, CU, ED = sys.argv[2:5]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
KTN = {("benchmark", "1.0"): "lai/benchmark_b1moi_lai_T10_seed20261107_20261114.json", ("benchmark", "0.6"): "lai/benchmark_b1moi_lai_T06_seed20261107_20261114.json",
       ("beta", "1.0"): "lai_beta/beta_b1moi_lai-xn_T10_seed20261107_20261114.json", ("beta", "0.6"): "lai_beta/beta_b1moi_lai-xn_T06_seed20261107_20261114.json"}
tb_cong, dat_cong = cong(os.path.join(ED, "du_benchmark_T10"))                  # E1 bat buoc cong tai hien dat
out = dict(seed=SEED, bo={}, cong_tai_hien=dict(tb=tb_cong, dat=dat_cong, khoang=[0.72, 0.86]))
for ds in ("benchmark", "beta"):
    out["bo"][ds] = {}
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        kj = json.load(open(os.path.join(CU, KTN[(ds, T)]), encoding="utf-8")); k = kj["tung_nguoi"]
        assert kj["ds"] == ds and abs(kj["T"] - float(T)) < 1e-9 and kj["seed"] == SEED and kj["n0"] == 40 and kj["nch"] == 22, KTN[(ds, T)]
        assert len(k) == SO[ds] and [p["fold"] for p in k] == list(range(1, SO[ds] + 1))
        ktn = np.array([p["B"]["SF"]["sau"] for p in k])
        assert np.isfinite(ktn).all() and (ktn >= 0).all() and (ktn <= 1).all()
        e = dict(tb={"KTN": float(100 * ktn.mean())})
        for bt in ("22", "du"):
            d = os.path.join(ED, "%s_%s_%s" % (bt, ds, tt))
            if not os.path.isdir(d):
                continue
            J = nap_dir(d, ds, bt, float(T))
            x = np.array([j["sau"] for j in J])
            e["tb"]["ED_" + bt] = float(100 * x.mean()); e["tb"]["ED_" + bt + "_phien"] = float(100 * np.mean([j["phien"] for j in J]))
            e["tb"]["giai_doan1_" + bt] = float(100 * np.mean([j["mang_giai_doan1_sau"] for j in J])); e["k_tb_" + bt] = float(np.mean([j["k_tb_sau"] for j in J]))
            e["KTN_tru_ED_" + bt] = ci(ktn - x)
            e["_x_" + bt] = x
        if "_x_du" in e and "_x_22" in e:
            e["ED_du_tru_ED_22"] = ci(e["_x_du"] - e["_x_22"])
        for kk in [kk for kk in e if kk.startswith("_x_")]:
            e.pop(kk)
        out["bo"][ds][T] = e
        print(ds, T, {a: round(b, 1) for a, b in e["tb"].items()}, {a: round(b, 1) for a, b in e.items() if a.startswith("k_tb")})
        print("    ", " | ".join("%s %s" % (a, f(b)) for a, b in e.items() if isinstance(b, dict) and "thap" in b))
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
