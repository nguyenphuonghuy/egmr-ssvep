#!/usr/bin/env python
"""Phan tich DH-TT.
Dung: python dh_tt_phan_tich.py <json ra> <thu muc ket-qua dh_tt> <thu muc ket-qua-b2e4r>
Ghep theo nguoi (fold): tep dh_tt {bo}_dh-tt_T{10|06}_*.json voi tep EGM-R {bo}_b2e4r_T{..}_*.json (Dong o thu muc con dong/).
Kiem ghep: UMR cua tep dh_tt trung KTNR cua tep b2e4r tung nguoi.
Phep so: chenh ghep cap theo nguoi, BCa 10.000, seed 20261107 (percentile neu suy bien, ghi ro); moi bo, moi cua so rieng.
  EGM-R - A (alpha 1e-5): can duoi > 0 'hon', can tren < 0 'kem', con lai 'chua phan biet'. Bao tach UMR - A0 va (EGM-R - UMR) so (A - A0).
  A* va C*: alpha tot nhat trong {1e-5, 1e-4, 1e-3} theo TB nguoi cho moi (bo, cua so), hoa lay alpha nho; bao TB va CI, KHONG xep hon/kem.
  P4_DH (A so A0, luot 41..) < 0,02: ghi 'quyet dinh cua A gan nhu trung A0 voi sieu tham so cua bai'.
"""
import glob, hashlib, json, os, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
RA, DH, B2 = sys.argv[1:4]
if os.path.exists(RA):
    raise SystemExit("tep ket qua da co: khong ghi de")
N = {"benchmark": 35, "beta": 70, "dong": 59}
AL = ("1e-05", "0.0001", "0.001")


def ci(a):
    a = 100 * np.asarray(a, float)
    assert np.isfinite(a).all(), "gia tri khong huu han"
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


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def xep(c):
    return "hon" if c["thap"] > 0 else ("kem" if c["cao"] < 0 else "chua phan biet")


def mot(pat):
    fs = sorted(f for f in glob.glob(pat) if not f.endswith(".tmp"))
    assert len(fs) == 1, (pat, fs)
    return fs[0]


out = dict(seed=SEED, tep={}, o={})
for ds in ("benchmark", "beta", "dong"):
    n = N[ds]
    for T, tt in (("1.0", "T10"), ("0.6", "T06")):
        fd = mot(os.path.join(DH, f"{ds}_dh-tt_{tt}_seed{SEED}_*.json"))
        fe = mot(os.path.join(B2, "dong" if ds == "dong" else "", f"{ds}_b2e4r_{tt}_seed{SEED}_*.json"))
        J = json.load(open(fd, encoding="utf-8")); Eg = json.load(open(fe, encoding="utf-8"))
        assert J["du_fold"] and Eg["du_fold"] and J["ds"] == Eg["ds"] == ds
        assert abs(J["T"] - float(T)) < 1e-9 and abs(Eg["T"] - float(T)) < 1e-9
        P, Q = J["tung_nguoi"], Eg["tung_nguoi"]
        assert [p["fold"] for p in P] == [q["fold"] for q in Q] == list(range(1, n + 1))
        umr = np.array([p["UMR"] for p in P]); assert np.allclose(umr, [q["KTNR"] for q in Q], atol=1e-12), "ghep nguoi lech"
        out["tep"][os.path.basename(fd)] = sha(fd); out["tep"][os.path.basename(fe)] = sha(fe)
        v = lambda nh, al, k="acc": np.array([p[nh][al][k] for p in P])
        egm = np.array([q["E4Rb"] for q in Q]); egml = np.array([q["E4Ra"] for q in Q]); w0 = np.array([p["W0"] for p in P])
        if all("tinh" in q for q in Q):                                  # Dong: doi chieu W0 voi tep EGM-R
            assert np.allclose(w0, [q["tinh"] for q in Q], atol=1e-12), "W0 lech tep EGM-R"
        assert all(p["A"][al]["bo_khong_huu_han"] == 0 for p in P for al in AL), "co luot bo cap nhat do gradient khong huu han: kiem tay"
        A0, A = v("A", "0.0"), v("A", "1e-05")
        mA = {al: v("A", al).mean() for al in AL}; aA = min(AL, key=lambda al: (-round(mA[al], 12), float(al)))
        mC = {al: v("C", al).mean() for al in AL}; aC = min(AL, key=lambda al: (-round(mC[al], 12), float(al)))
        As, Cs, C5 = v("A", aA), v("C", aC), v("C", "1e-05")
        e = dict(n=n, tb={k: 100 * float(x.mean()) for k, x in dict(W0=w0, UMR=umr, EGM_R=egm, EGM_Rl=egml, A0=A0, A=A, Astar=As,
                                                                  C0=v("C", "0.0"), C5=C5, Cstar=Cs).items()},
                 alpha_Astar=aA, alpha_Cstar=aC, tb_A_moi_alpha={al: 100 * float(mA[al]) for al in AL},
                 tb_C_moi_alpha={al: 100 * float(mC[al]) for al in AL})
        so = {"EGM-A": egm - A, "UMR-A0": umr - A0, "EGM-UMR": egm - umr, "A-A0": A - A0, "A0-W0": A0 - w0,
              "EGM-A*": egm - As, "A*-A": As - A, "C(1e-5)-UMR": C5 - umr, "C*-UMR": Cs - umr, "EGM-C*": egm - Cs, "EGMl-C*": egml - Cs}
        e["so"] = {k: ci(x) for k, x in so.items()}
        e["so"]["EGM-A"]["xep"] = xep(e["so"]["EGM-A"])
        for k in ("EGM-A*", "EGM-C*", "EGMl-C*", "C*-UMR", "A*-A"):
            e["so"][k]["ghi_chu"] = "alpha chon bang nhan dich (oracle): muc lac quan, khong xep hon/kem"
        cc = lambda nh, al, k0: dict(**{k0: float(v(nh, al, k0).mean())}, chuan_tb=float(v(nh, al, "chuan_tb").mean()),
                                     dich_logit_tb=float(v(nh, al, "dich_logit_tb").mean()), ti_le_vuot_gamma_tb=float(v(nh, al, "ti_le_vuot_gamma").mean()),
                                     F_max_tb=float(v(nh, al, "F_max").mean()), bo=int(v(nh, al, "bo_khong_huu_han").sum()))
        e["co_che"] = {al: cc("A", al, "P4_so_A0") for al in AL}
        e["co_che_C"] = {al: cc("C", al, "P4_so_C0") for al in AL}
        e["ci_do_chinh_xac"] = dict(A=ci(A), Astar=ci(As), Cstar=ci(Cs), ghi_chu="Astar, Cstar: alpha chon bang nhan dich (muc lac quan)")
        e["cong_C0"] = dict(so_quyet_dinh_khac_UMR=int(sum(p["cong_C0"]["so_quyet_dinh_khac_UMR"] for p in P)),
                            lech_diem_toi_da=float(max(p["cong_C0"]["lech_diem_toi_da"] for p in P)))
        e["A_ca_phien"] = {al: 100 * float(v("A", al, "acc_ca_phien").mean()) for al in ("0.0",) + AL}
        p4 = e["co_che"]["1e-05"]["P4_so_A0"]
        e["cau"] = dict(EGM_A="EGM-R %s A (ban chuyen DH-TT sang mang va tien xu ly B2): %+.2f [%+.2f; %+.2f]" % (
            {"hon": "hon", "kem": "kem", "chua phan biet": "chua phan biet duoc voi"}[e["so"]["EGM-A"]["xep"]],
            e["so"]["EGM-A"]["tb"], e["so"]["EGM-A"]["thap"], e["so"]["EGM-A"]["cao"]),
            tach="diem xuat phat UMR - A0 %+.2f; thich nghi: EGM-R - UMR %+.2f so voi A - A0 %+.2f" % (
                e["so"]["UMR-A0"]["tb"], e["so"]["EGM-UMR"]["tb"], e["so"]["A-A0"]["tb"]),
            P4=("quyet dinh cua A gan nhu trung A0 voi sieu tham so cua bai (P4_DH = %.4f)" % p4) if p4 < 0.02 else ("P4_DH = %.4f" % p4))
        out["o"][f"{ds}_{T}"] = e
        print(ds, T, "| A %.2f (alpha* %s) | C* %s |" % (e["tb"]["A"], aA, aC), e["cau"]["EGM_A"], "|", e["cau"]["P4"], flush=True)
json.dump(out, open(RA + ".tmp", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
os.link(RA + ".tmp", RA); os.remove(RA + ".tmp")                              # khong ghi de
