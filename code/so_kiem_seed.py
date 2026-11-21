"""So ket qua cua job kiem b2seed_kiem.pbs voi ket qua da luu (chay trong job, can torch). Thoat ma 1 neu bat ky phep kiem nao lech.
 K1: trong so BETA fold 1 huan luyen lai (seed mac dinh) trung tung bit voi results/beta_nen/T10/fold001_trongso.pt.
 K2: b2e4r --bo-json-goc (seed mac dinh) Benchmark fold 1-2: quyet dinh KTNR, E4Ra, E4Rb va do chinh xac trung JSON b2e4r da luu.
 K3: ktnr_buoc1 (seed mac dinh) Benchmark 2 fold: moi truong so cua tung fold trung JSON KTN-R da luu (bo qua sha, thoi gian, duong dan).
 K4: seed moi Dong fold 1 chay thong: co trong so va JSON, gia tri huu han; seed ghi trong JSON la seed moi. Khong in do chinh xac.
Dung: python so_kiem_seed.py <thu muc kiem>"""
import json, math, os, sys
import torch
O = sys.argv[1]
loi = []


def kiem(ten, dk):
    print(("DAT " if dk else "LECH"), ten, flush=True)
    if not dk:
        loi.append(ten)


a = torch.load(os.path.join(O, "nen_beta_f1", "fold001_trongso.pt"), map_location="cpu")
b = torch.load("results/beta_nen/T10/fold001_trongso.pt", map_location="cpu")
kiem("K1 trong so BETA fold 1", a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a))

m = json.load(open(os.path.join(O, "benchmark_b2e4r_f12.json")))
g = json.load(open("results/b2e4r/benchmark_b2e4r_T10_seed20261107_20261117.json"))
for u in m["tung_nguoi"]:
    v = g["tung_nguoi"][u["fold"] - 1]
    kiem("K2 b2e4r fold %d" % u["fold"], all(u["dec"][k] == v["dec"][k] for k in ("KTNR", "E4Ra", "E4Rb"))
         and all(u[k] == v[k] for k in ("KTNR", "E4Rb", "E4Ra", "nguon_chon5", "seed")))

BO = ("sha256", "giay", "thoi_gian", "trongso", "duong_dan", "tep")
k = json.load(open(os.path.join(O, "benchmark_ktnr_f12.json")))
kg = json.load(open("results/b2e4r_goc/benchmark_ktnr-buoc1_T10_seed20261107_20261115.json"))


def bang(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or x == y
    return x == y


for u in k["tung_nguoi"]:
    v = kg["tung_nguoi"][u["fold"] - 1]
    khoa = [q for q in v if not any(z in q for z in BO)]
    kiem("K3 ktnr fold %d (%d truong)" % (u["fold"], len(khoa)), all(q in u and bang(u[q], v[q]) for q in khoa))

d = json.load(open(os.path.join(O, "dong_ktnr_s08_f1.json")))
ok = os.path.exists(os.path.join(O, "nen_dong_s08_f1", "fold001_trongso.pt")) and d["seed"] == 20261108 and len(d["tung_nguoi"]) == 1
ok = ok and all(math.isfinite(x) for x in d["tung_nguoi"][0].values() if isinstance(x, float))
kiem("K4 seed moi Dong fold 1 chay thong", ok)
print("TONG LECH:", len(loi), loi)
sys.exit(1 if loi else 0)
