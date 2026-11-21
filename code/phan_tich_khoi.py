"""Phan tich theo khoi va tom tat theo nguoi cho EGM-R, tu ket qua da luu (khong chay mo hinh).
 (a) hieu EGM-R (E4Rb) - UM-R (KTNR) theo khoi 40 luot sau khoi dong, va chenh khoi cuoi - khoi dau;
 (b) khoang BCa hieu chinh Bonferroni (98,33 %) cho ba so sanh chinh o 1,0 s, kem Wilcoxon co Holm;
 (c) so luot moi nguoi duoc sua dung (UM-R sai, EGM-R dung) va bi lam sai (nguoc lai);
 (d) ITR tuyet doi cua UM-R va EGM-R (tu phan-bo-itr_20261120.json);
 (e) EGM-R - EGM-R_40 (A2 trong b2e4r-dclam) theo khoi.
Nhan luot i trong "dec" (luot tuyet doi 40 + i): (40 + i) mod 40, kiem bang cach tinh lai do chinh xac da luu.
BCa 10 000 lan, seed 20261107. Dung: python phan_tich_khoi.py <json ra>"""
import json, os, sys
import numpy as np
from scipy.stats import bootstrap, wilcoxon
sys.stdout.reconfigure(encoding="utf-8")
SEED = 20261107
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ket-qua-b2e4r") + "/"
TEP = {("benchmark", "10"): R + "benchmark_b2e4r_T10_seed20261107_20261117.json", ("benchmark", "06"): R + "benchmark_b2e4r_T06_seed20261107_20261117.json",
       ("beta", "10"): R + "beta_b2e4r_T10_seed20261107_20261117.json", ("beta", "06"): R + "beta_b2e4r_T06_seed20261107_20261117.json",
       ("dong", "10"): R + "dong/dong_b2e4r_T10_seed20261107_20261117.json", ("dong", "06"): R + "dong/dong_b2e4r_T06_seed20261107_20261117.json"}
ITR = R + "bo-sung-20261120/phan-bo-itr_20261120.json"


def ci(a, muc=0.95):
    a = np.asarray(a, float)
    tb = float(a.mean())
    if np.allclose(a, a[0]):
        return dict(tb=tb, thap=None, cao=None)
    r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", confidence_level=muc, random_state=np.random.default_rng(SEED)).confidence_interval
    return dict(tb=tb, thap=float(r.low), cao=float(r.high))


out = dict(seed=SEED, ghi_chu="don vi diem phan tram tru khi ghi khac; khoi = 40 luot lien tiep sau luot 40", khoi={}, chinh_bonferroni={}, sua_hong={}, itr={})
p_chinh = {}
for (ds, w), f in TEP.items():
    J = json.load(open(f, encoding="utf-8"))
    tn = J["tung_nguoi"]
    n = tn[0]["n_cham"]; y = (40 + np.arange(n)) % 40
    for u in tn:                                                    # kiem gia dinh nhan
        for k in ("KTNR", "E4Rb"):
            assert abs(float((np.array(u["dec"][k]) == y).mean()) - u[k]) < 1e-12, (ds, w, u["fold"], k)
    K = np.array([u["dec"]["KTNR"] for u in tn]) == y; E = np.array([u["dec"]["E4Rb"] for u in tn]) == y
    nk = n // 40
    d_khoi = [100 * (E[:, 40 * b:40 * (b + 1)].mean(1) - K[:, 40 * b:40 * (b + 1)].mean(1)) for b in range(nk)]
    out["khoi"][f"{ds}_{w}"] = dict(so_khoi=nk, hieu_theo_khoi=[ci(d) for d in d_khoi], cuoi_tru_dau=ci(d_khoi[-1] - d_khoi[0]))
    sua = (E & ~K).sum(1); hong = (~E & K).sum(1)
    out["sua_hong"][f"{ds}_{w}"] = dict(n_nguoi=len(tn), n_luot=int(n), sua_tb=float(sua.mean()), hong_tb=float(hong.mean()), rong=ci(sua - hong),
                                        ti_le_nguoi_hong_lon_hon_sua=float((hong > sua).mean()))
    if w == "10":
        d = 100 * (E.mean(1) - K.mean(1))
        out["chinh_bonferroni"][ds] = dict(ci95=ci(d), ci9833=ci(d, 1 - 0.05 / 3))
        p_chinh[ds] = float(wilcoxon(d, zero_method="wilcox").pvalue)
# Holm tren p Wilcoxon cua ba so sanh chinh
thu = sorted(p_chinh, key=p_chinh.get); m = len(thu); dieu = {}; cur = 0.0
for i, ds in enumerate(thu):
    cur = max(cur, min(1.0, (m - i) * p_chinh[ds])); dieu[ds] = cur
for ds in p_chinh:
    out["chinh_bonferroni"][ds].update(p_wilcoxon=p_chinh[ds], p_holm=dieu[ds])
# (e) EGM-R_40 (A2 trong b2e4r-dclam: khop lo tren 40 luot khoi dong, giu co dinh tu luot 41) theo khoi
out["khoi_mot_lan"] = {}
for (ds, w) in TEP:
    f2 = R + "bo-sung-20261120/%s_b2e4r-dclam_T%s_seed20261107_20261120.json" % (ds, w)
    tn = json.load(open(f2, encoding="utf-8"))["tung_nguoi"]
    n = tn[0]["n_cham"]; y = (40 + np.arange(n)) % 40; nk = n // 40
    for u in tn:
        assert abs(float((np.array(u["dec"]["A2"]) == y).mean()) - u["A2"]) < 1e-12, (ds, w, u["fold"])
    A2 = np.array([u["dec"]["A2"] for u in tn]) == y
    tb = {u["fold"]: u for u in json.load(open(TEP[(ds, w)], encoding="utf-8"))["tung_nguoi"]}
    E = np.array([tb[u["fold"]]["dec"]["E4Rb"] for u in tn]) == y; K = np.array([tb[u["fold"]]["dec"]["KTNR"] for u in tn]) == y
    kh = lambda X, Y: [100 * (X[:, 40 * b:40 * (b + 1)].mean(1) - Y[:, 40 * b:40 * (b + 1)].mean(1)) for b in range(nk)]
    dE, dA = kh(E, A2), kh(A2, K)
    out["khoi_mot_lan"][f"{ds}_{w}"] = dict(egmr_tru_egmr40=[ci(d) for d in dE], egmr40_tru_umr=[ci(d) for d in dA],
                                            egmr_tru_egmr40_cuoi_tru_dau=ci(dE[-1] - dE[0]), toan_phien_egmr_tru_egmr40=ci(100 * (E.mean(1) - A2.mean(1))))
I = json.load(open(ITR, encoding="utf-8"))
for k, v in I.items():
    out["itr"][k] = dict(umr=v["itr_umr"], egmr=v["itr_egmr"], hieu=v["itr_hieu"], hieu_tuong_doi_phan_tram=100 * (v["itr_egmr"] / v["itr_umr"] - 1))
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def f(c):
    return "%+.2f [%s, %s]" % (c["tb"], "%+.2f" % c["thap"] if c["thap"] is not None else "-", "%+.2f" % c["cao"] if c["cao"] is not None else "-")


for k, v in out["khoi"].items():
    print(k, "theo khoi:", " | ".join(f(c) for c in v["hieu_theo_khoi"]), "|| cuoi-dau", f(v["cuoi_tru_dau"]))
for k, v in out["chinh_bonferroni"].items():
    print("chinh", k, "95%", f(v["ci95"]), "98,33%", f(v["ci9833"]), "p W %.2g Holm %.2g" % (v["p_wilcoxon"], v["p_holm"]))
for k, v in out["sua_hong"].items():
    print("sua/hong", k, "sua %.2f hong %.2f rong %s ti le hong>sua %.2f" % (v["sua_tb"], v["hong_tb"], f(v["rong"]), v["ti_le_nguoi_hong_lon_hon_sua"]))
for k, v in out["khoi_mot_lan"].items():
    print("EGM-R - EGM-R_40", k, " | ".join(f(c) for c in v["egmr_tru_egmr40"]), "|| cuoi-dau", f(v["egmr_tru_egmr40_cuoi_tru_dau"]),
          "|| ca phien", f(v["toan_phien_egmr_tru_egmr40"]))
    print("EGM-R_40 - UM-R  ", k, " | ".join(f(c) for c in v["egmr40_tru_umr"]))
for k, v in out["itr"].items():
    print("itr", k, "UM-R %.1f EGM-R %.1f (+%.1f %%)" % (v["umr"], v["egmr"], v["hieu_tuong_doi_phan_tram"]))
