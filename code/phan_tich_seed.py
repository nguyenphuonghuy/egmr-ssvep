"""Phep so chinh EGM-R (E4Rb) - UM-R (KTNR) o 1,0 s theo seed huan luyen, tu cac JSON b2e4r da luu.
Seed 20261107: ket qua cua bai (results/ket-qua-b2e4r/, dong/). Seed 20261108, 20261109: results/ket-qua-b2e4r/seed/.
Voi moi (bo, seed): TB UM-R, TB EGM-R, hieu 100 * (E4Rb - KTNR) theo nguoi, CI 95 % BCa (cung ham ci voi b2e4r_phan_tich.py:
10 000 lan, seed 20261107), DCR, ti le nguoi kem di. Them: hieu TB qua ba seed theo tung nguoi (CI BCa qua nguoi), do lech chuan
giua seed cua hieu TB va cua do chinh xac UM-R.
Dung: python phan_tich_seed.py <thu muc results/ket-qua-b2e4r> <json ra>"""
import glob, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
G = sys.argv[1].rstrip("/\\") + "/"
GOC = {"benchmark": G + "benchmark_b2e4r_T10_seed20261107_20261117.json", "beta": G + "beta_b2e4r_T10_seed20261107_20261117.json",
       "dong": G + "dong/dong_b2e4r_T10_seed20261107_20261117.json"}


SO_NGUOI = {"benchmark": 35, "beta": 70, "dong": 59}
SEEDS = [20261107, 20261108, 20261109]


def ci(a):
    """Cung cach voi b2e4r_phan_tich.ci (dau vao da o don vi diem %): BCa 10 000 lan, seed 20261107; suy bien -> None."""
    a = np.asarray(a, float)
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


def doc(f, ds):
    J = json.load(open(f, encoding="utf-8"))
    assert abs(J["T"] - 1.0) < 1e-9 and J["ds"] == ds, f"{f}: sai bo hoac cua so"
    assert J["alpha"] == 0.5 and J["H"] == 5 and J["k"] == 5 and J["n0"] == 40 and J.get("du_fold") is True, f"{f}: cau hinh khac hoac chay thu"
    tn = sorted(J["tung_nguoi"], key=lambda u: u["fold"])
    assert [u["fold"] for u in tn] == list(range(1, SO_NGUOI[ds] + 1)), f"{f}: thieu, trung hoac sai thu tu fold"
    assert all(u["seed"] == J["seed"] + 1000 * u["fold"] + 10 for u in tn), f"{f}: seed theo fold lech"
    assert all(np.isfinite([u["KTNR"], u["E4Rb"], u["P4_b"]]).all() for u in tn), f"{f}: chi so khong huu han"
    k = np.array([u["KTNR"] for u in tn]); e = np.array([u["E4Rb"] for u in tn]); p = np.array([u["P4_b"] for u in tn])
    return int(J["seed"]), [u["fold"] for u in tn], k, e, p


out = {"seed_bca": SEED, "bo": {}}
for ds in ("benchmark", "beta", "dong"):
    tep = [GOC[ds]] + sorted(glob.glob(G + f"seed/{ds}_b2e4r_T10_seed*_*.json"))
    theo = {}
    for f in tep:
        sd, fo, k, e, p = doc(f, ds)
        assert sd not in theo, (ds, sd)
        theo[sd] = dict(fold=fo, k=k, e=e, p=p, tep=os.path.basename(f))
    assert sorted(theo) == SEEDS, f"{ds}: can dung ba seed {SEEDS}, co {sorted(theo)}"
    o = {"seed": {}}
    for sd, v in sorted(theo.items()):
        d = 100 * (v["e"] - v["k"])
        o["seed"][str(sd)] = dict(n=len(d), umr=100 * float(v["k"].mean()), egmr=100 * float(v["e"].mean()), hieu=ci(d), dcr=float(v["p"].mean()),
                                  ti_le_kem=float((d < 0).mean()), ti_le_hon=float((d > 0).mean()), tep=v["tep"])
    if len(theo) > 1:
        fo0 = theo[min(theo)]["fold"]
        assert all(v["fold"] == fo0 for v in theo.values()), ds
        H = np.array([100 * (v["e"] - v["k"]) for v in theo.values()])
        o["tb_qua_seed"] = dict(so_seed=len(theo), hieu=ci(H.mean(0)), sd_giua_seed_cua_hieu_tb=float(np.std([h.mean() for h in H], ddof=1)),
                                sd_giua_seed_cua_umr=float(np.std([100 * v["k"].mean() for v in theo.values()], ddof=1)),
                                lech_umr_max_min=float(np.ptp([100 * v["k"].mean() for v in theo.values()])))
    out["bo"][ds] = o
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for ds, o in out["bo"].items():
    for sd, v in o["seed"].items():
        h = v["hieu"]
        print(f"{ds:9s} seed {sd}: UM-R {v['umr']:.2f} EGM-R {v['egmr']:.2f} hieu {h['tb']:+.2f} [{h['thap']:+.2f}; {h['cao']:+.2f}] DCR {v['dcr']:.3f} kem {v['ti_le_kem']:.2f}")
    if "tb_qua_seed" in o:
        t = o["tb_qua_seed"]; h = t["hieu"]
        print(f"{ds:9s} TB {t['so_seed']} seed: hieu {h['tb']:+.2f} [{h['thap']:+.2f}; {h['cao']:+.2f}]; SD giua seed: hieu {t['sd_giua_seed_cua_hieu_tb']:.2f}, UM-R {t['sd_giua_seed_cua_umr']:.2f}")
