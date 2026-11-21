#!/usr/bin/env python
"""Phan tich tai hien CCA-Net.
  cong <dir goc benchmark T10> <dir goc beta T10> [cong.json] :
      CN-goc, bien the 'final' (CCA-Net THEO MA commit 1660d42), cham MOI luot, TB qua nguoi; dat khi Benchmark 1,0 s trong
      [80,02; 88,02] VA BETA 1,0 s trong [68,01; 76,01] (Bang I cua bai: 84,02 / 72,01 %, +-4 diem; tac gia sua cong 2026-11-16,
      thay cong cu 79,0 / 73,9 doi tu ITR trung binh). Ma thoat 0 dat, 5 khong dat.
  bao <json ra> <thu muc ket-qua-b1moi> <thu muc ban sao results/b1moi_ccanet> :
      Benchmark, BETA x 1,0 / 0,6 s, cham luot 41 den het: KTN - CN-gt, KTN-R - CN-gt (CI 95 % BCa 10.000 lan, seed 20261107), TB cac thanh phan
  E2 (Nakanishi) them vao tep nay khi ma buoc 2 KTN-R co dinh dang, TRUOC khi chay Nakanishi.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
SO = {"nak": 10, "benchmark": 35, "beta": 70}


def nap(d, ds, T, che):
    cfg = json.load(open(os.path.join(d, "config.json")))
    assert cfg["ds"] == ds and abs(cfg["T"] - T) < 1e-9 and cfg["che"] == che and not cfg["du_nhanh"] and cfg["vong"] == 100, (d, cfg)
    assert cfg["commit"] == "1660d42113d5721c5ec05d5f8b92aeabcd50c31f"
    J = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "fold*.json")))]
    assert [j["fold"] for j in J] == list(range(1, SO[ds] + 1)), (d, len(J))
    assert len({j["script_sha256"] for j in J}) == 1 and J[0]["script_sha256"] == cfg["script_sha256"]
    fs_ = {"nak": 256, "benchmark": 250, "beta": 250}[ds]; ns_ = int(T * fs_) if che == "goc" else int(round(T * fs_))
    assert cfg["ns"] == ns_ and cfg["rfs"] == fs_ and cfg["seed"] == (41 if che == "goc" else SEED), d
    for j in J:
        assert j["du_nhanh"] is False and j["ds"] == ds and j["che"] == che and abs(j["T"] - T) < 1e-9 and j["ns"] == ns_, (d, j["fold"])
        assert j["seed"] == (41 if che == "goc" else SEED + 1000 * j["fold"] + int(round(T * 10))), (d, j["fold"])
        assert sum(j["luot_vecto_hang"][k] for k in ("CNN", "f1", "f2", "f3")) == 0, ("vecto diem hang", d, j["fold"])
    for j in J:
        for k in ("final", "CNN", "FBTTCCA_new+CCA_new"):
            for t in ("moi_luot", "sau"):
                v = j[k][t]; assert np.isfinite(v) and 0 <= v <= 1, (d, j["fold"], k)
    return J


def ci(a):
    a = 100 * np.asarray(a, float)
    if np.allclose(a, a[0]):
        return dict(tb=float(a.mean()), thap=float(a.mean()), cao=float(a.mean()), phuong_phap="hang")
    pp = "BCa"
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            raise ValueError
    except Exception:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="percentile", random_state=np.random.default_rng(SEED)).confidence_interval; pp = "percentile"
    return dict(tb=float(a.mean()), thap=float(r.low), cao=float(r.high), phuong_phap=pp)


if sys.argv[1] == "cong":
    b = np.mean([j["final"]["moi_luot"] for j in nap(sys.argv[2], "benchmark", 1.0, "goc")]) * 100
    e = np.mean([j["final"]["moi_luot"] for j in nap(sys.argv[3], "beta", 1.0, "goc")]) * 100
    dat = bool(80.02 <= b <= 88.02 and 68.01 <= e <= 76.01)
    print("CONG TAI HIEN CCA-Net: Benchmark 1,0 s %.2f (84,02 +- 4), BETA 1,0 s %.2f (72,01 +- 4):" % (b, e), "DAT" if dat else "KHONG DAT", flush=True)
    if len(sys.argv) > 4:                                                     # ghi cong.json cho ktnr_buoc2_phan_tich.py (E2)
        json.dump(dict(benchmark_T10=float(b), beta_T10=float(e), dat=dat), open(sys.argv[4], "w"), indent=1)
    sys.exit(0 if dat else 5)

RA, CU, CN = sys.argv[2:5]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
KTN = {("benchmark", "1.0"): "lai/benchmark_b1moi_lai_T10_seed20261107_20261114.json", ("benchmark", "0.6"): "lai/benchmark_b1moi_lai_T06_seed20261107_20261114.json",
       ("beta", "1.0"): "lai_beta/beta_b1moi_lai-xn_T10_seed20261107_20261114.json", ("beta", "0.6"): "lai_beta/beta_b1moi_lai-xn_T06_seed20261107_20261114.json"}
out = dict(seed=SEED, bo={})
try:
    out["cong"] = dict(benchmark_T10=100 * float(np.mean([j["final"]["moi_luot"] for j in nap(os.path.join(CN, "goc_benchmark_T10"), "benchmark", 1.0, "goc")])),
                       beta_T10=100 * float(np.mean([j["final"]["moi_luot"] for j in nap(os.path.join(CN, "goc_beta_T10"), "beta", 1.0, "goc")])))
    out["cong"]["dat"] = bool(80.02 <= out["cong"]["benchmark_T10"] <= 88.02 and 68.01 <= out["cong"]["beta_T10"] <= 76.01)
except (AssertionError, FileNotFoundError) as ex:
    out["cong"] = dict(dat=False, loi=str(ex)[:200])
for ds in ("benchmark", "beta"):
    out["bo"][ds] = {}
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        d = os.path.join(CN, "gt_%s_%s" % (ds, tt))
        if not os.path.isdir(d):
            print("CANH BAO: thieu", d); out["bo"][ds][T] = "thieu"; continue
        J = nap(d, ds, float(T), "gt")
        kj = json.load(open(os.path.join(CU, KTN[(ds, T)]), encoding="utf-8"))["tung_nguoi"]
        assert len(kj) == SO[ds] and [p["fold"] for p in kj] == list(range(1, SO[ds] + 1))
        ktn = np.array([p["B"]["SF"]["sau"] for p in kj])
        cn = np.array([j["final"]["sau"] for j in J])
        e = dict(tb=dict(KTN=100 * float(ktn.mean()), CN_gt=100 * float(cn.mean()), CNN=100 * float(np.mean([j["CNN"]["sau"] for j in J])),
                         MFBtt=100 * float(np.mean([j["FBTTCCA_new+CCA_new"]["sau"] for j in J]))), KTN_tru_CN=ci(ktn - cn))
        kr = glob.glob(os.path.join(CU, "ktnr", "%s_ktnr-buoc1_%s_seed%d_*.json" % (ds, tt, SEED)))
        if len(kr) != 1:
            print("CANH BAO: tep KTN-R %s %s %s (can dung 1)" % (ds, tt, "thieu" if not kr else "khong duy nhat")); e["KTN_R"] = "thieu hoac khong duy nhat"
        else:
            P = json.load(open(kr[0], encoding="utf-8"))["tung_nguoi"]; assert [p["fold"] for p in P] == list(range(1, SO[ds] + 1))
            x = np.array([p["ktnr"] for p in P]); e["tb"]["KTN_R"] = 100 * float(x.mean()); e["KTNR_tru_CN"] = ci(x - cn)
        out["bo"][ds][T] = e
        print(ds, T, {k: round(v, 1) for k, v in e["tb"].items()}, "| KTN-CN %+.2f [%+.2f; %+.2f]" % (e["KTN_tru_CN"]["tb"], e["KTN_tru_CN"]["thap"], e["KTN_tru_CN"]["cao"]))
print("cong tai hien:", out["cong"])
json.dump(out, open(RA, "w", encoding="utf-8"), indent=1)
