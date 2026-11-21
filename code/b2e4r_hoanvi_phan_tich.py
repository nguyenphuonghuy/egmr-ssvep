#!/usr/bin/env python
"""Phan tich do ben E4R-b.
Ban sua tu b1gop_hoanvi_phan_tich.py (thi nghiem A): cap (E4, E0) thay bang (E4Rb, KTNR).
Moi o (bo, cua so, ho): Delta_ir = Acc(E4Rb) - Acc(KTNR) sau luot 40 theo hoan vi r (r = 1..R; r = 0 la dong nhat, KHONG gop vao TB);
d_i = TB_r Delta_ir; D = TB_i d_i; Delta_i0 = hieu o r = 0 (thu tu tep); q_i = d_i - Delta_i0, Q = TB_i q_i. Don vi: diem phan tram.
CI 95 % BCa qua nguoi, 10.000 lan, seed 20261107; CI khong huu han hoac mau hang (SE = 0): "khong tinh duoc".
MCSE(D) = sqrt(sum_i s_i^2 / R) / n, s_i^2 = phuong sai mau theo r cua Delta_ir; SE(d_i) = do lech chuan mau cua d_i / sqrt(n).
Luat tang R (chi xet o P1, hai bo x hai cua so): MCSE(D) > 0,25 SE(d_i) o bat ky o nao -> tang_R = true (chay them r = 21..40 cho moi o P1 va P2).
Trang thai H8 tung bo (1,0 s chinh; 0,6 s bao kem cung quy tac): duong (thap > 0), khong duong (cao <= 0), chua phan biet (CI chua 0).
Q mot phia: thap(Q) > -0,5 tren bo nao -> "khong giam qua 0,5 diem" cho bo do.
Dung: python b2e4r_hoanvi_phan_tich.py <bm T10> <bm T06> <beta T10> <beta T06> <json ra> [tep mo rong r 21..40 theo cung thu tu, 4 tep]
"""
import json, sys
import numpy as np
from scipy.stats import bootstrap

SEED = 20261107
BIEN_Q = -0.5
CAU_HINH = [("benchmark", "1.0"), ("benchmark", "0.6"), ("beta", "1.0"), ("beta", "0.6")]
N_NGUOI = {"benchmark": 35, "beta": 70}


def ci(a):
    a = 100 * np.asarray(a, float)
    tb = float(a.mean())
    if not np.isfinite(a).all() or np.allclose(a, a[0]):
        return dict(tb=tb, thap=None, cao=None, ghi_chu="khong tinh duoc")
    try:
        r = bootstrap((a,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(SEED)).confidence_interval
    except Exception as e:                                                # BCa suy bien thi bao, khong sap
        return dict(tb=tb, thap=None, cao=None, ghi_chu="khong tinh duoc: " + str(e)[:80])
    if not (np.isfinite(r.low) and np.isfinite(r.high)):
        return dict(tb=tb, thap=None, cao=None, ghi_chu="khong tinh duoc")
    return dict(tb=tb, thap=float(r.low), cao=float(r.high))


def trang_thai(c):
    if c["thap"] is None:
        return "khong tinh duoc"
    if c["thap"] > 0:
        return "duong"
    if c["cao"] <= 0:
        return "khong duong"
    return "chua phan biet"


def doc(fn, ds, T, r_dau, r_cuoi):
    j = json.load(open(fn, encoding="utf-8"))
    assert j["ds"] == ds and abs(j["T"] - float(T)) < 1e-9 and j["r"] == [r_dau, r_cuoi], fn
    assert [p["fold"] for p in j["tung_nguoi"]] == list(range(1, N_NGUOI[ds] + 1)) and j.get("du_fold") is True, fn + ": thieu nguoi, sai thu tu hoac chay thu"
    assert j["n0"] == 40 and j["alpha"] == 0.5 and j["seed"] == SEED, fn + ": cau hinh khac"
    assert all(p["seed"] == SEED + 1000 * p["fold"] + int(round(float(T) * 10)) for p in j["tung_nguoi"]), fn + ": seed fold lech"
    return j


def bang(ps, P, rs):
    """Tra (n, R) Delta_ir va cac dai luong kem cho ho P, hoan vi rs."""
    D, P4, ac, Gc, so_lop, lop_max, hist = [], [], [], [], [], [], []
    for p in ps:
        hv = {(h["P"], h["r"]): h for h in p["hoan_vi"]}
        assert all((P, r) in hv for r in rs), f"fold {p['fold']}: thieu hoan vi P{P}"
        D.append([hv[(P, r)]["E4Rb"] - hv[(P, r)]["KTNR"] for r in rs])
        P4.append(np.mean([hv[(P, r)]["P4"] for r in rs]))
        ac.append(np.nan)                                                 # E4R-b khong ghi g_t
        Gc.append(np.mean([hv[(P, r)]["G"] for r in rs]))
        if P == 2:
            so_lop.append(np.mean([hv[(P, r)]["so_lop_40_dau"] for r in rs])); lop_max.append(np.mean([hv[(P, r)]["lop_40_dau_max"] for r in rs]))
            hist.append([hv[(P, r)]["histogram_40_dau"] for r in rs])
    return np.array(D), np.array(P4), np.array(ac), np.array(Gc), so_lop, lop_max, hist


def can_tang_R(goc):
    """Luat tang R tren tien to r = 1..20, chi o P1 (hai bo x hai cua so)."""
    for j in goc:
        Dl = bang(j["tung_nguoi"], 1, list(range(1, 21)))[0]
        n = Dl.shape[0]; di = Dl.mean(1)
        mcse = float(np.sqrt(Dl.var(1, ddof=1).sum() / 20) / n); se = float(di.std(ddof=1) / np.sqrt(n))
        if not (np.isfinite(mcse) and np.isfinite(se)):                   # chi gia tri khong huu han moi la "khong tinh duoc"
            return "khong tinh duoc (MCSE hoac SE khong huu han o %s %s)" % (j["ds"], j["T"])
        if mcse > 0.25 * se:                                              # SE = 0 va MCSE > 0 van phai tang R
            return True
    return False


def main():
    a = sys.argv[1:]
    assert len(a) in (5, 9), __doc__
    goc = [doc(a[k], ds, T, 0, 20) for k, (ds, T) in enumerate(CAU_HINH)]
    mr = [doc(a[5 + k], ds, T, 21, 40) for k, (ds, T) in enumerate(CAU_HINH)] if len(a) == 9 else None
    tang20 = can_tang_R(goc)                                              # quyet dinh tang R tinh tu tien to R = 20
    if mr:
        assert tang20 is True, "dot mo rong r 21..40 chi duoc nhan khi luat tang R (tinh o R = 20) thoa"
        for g_, m_ in zip(goc, mr):                                       # cung nguon goc
            for k_ in ("script_sha256", "b2e4r_sha256", "b1gop_sha256", "b1moi_lai_sha256", "global_dnn_sha256", "b1_diem_sha256", "du_lieu_lee_sha256",
                       "json_b2e4r_sha256", "json_ktnr_sha256",
                       "trongso", "trongso_fold_sha256", "ho"):
                assert g_[k_] == m_[k_], f"dot mo rong khac dot dau o {k_}"
    R = 40 if mr else 20
    rs = list(range(1, R + 1))
    out = dict(R=R, o={}, tang_R_tai_R20=tang20)
    for k, (ds, T) in enumerate(CAU_HINH):
        ps = goc[k]["tung_nguoi"]
        if mr:                                                            # ghep hoan vi mo rong theo nguoi
            for p, q in zip(ps, mr[k]["tung_nguoi"]):
                assert p["fold"] == q["fold"]; p["hoan_vi"] = p["hoan_vi"] + q["hoan_vi"]
        d0 = []
        for p in ps:
            h0 = [h for h in p["hoan_vi"] if h["r"] == 0]
            assert len(h0) == 1; d0.append(h0[0]["E4Rb"] - h0[0]["KTNR"])
        d0 = np.array(d0)
        for P in (1, 2):
            Dl, P4, ac, Gc, so_lop, lop_max, hist = bang(ps, P, rs)
            n = Dl.shape[0]; di = Dl.mean(1); qi = di - d0
            mcse = 100 * float(np.sqrt(Dl.var(1, ddof=1).sum() / R) / n)
            se_d = 100 * float(di.std(ddof=1) / np.sqrt(n)); se_q = 100 * float(qi.std(ddof=1) / np.sqrt(n))
            Dc, Qc = ci(di), ci(qi)
            o = dict(D=Dc, Q=Qc, trang_thai=trang_thai(Dc),
                     Q_khong_giam_qua_0_5=("khong tinh duoc" if Qc["thap"] is None else bool(Qc["thap"] > BIEN_Q)),
                     thu_tu_tep=ci(d0), MCSE_D=mcse, SE_d=se_d, SE_q=se_q, MCSE_tren_SE=(mcse / se_d if se_d > 0 else None),
                     ti_le_Delta_duong=float((Dl > 0).mean()), ti_le_nguoi_d_duong=float((di > 0).mean()),
                     phan_vi_5_Delta_trong_nguoi_TB=100 * float(np.percentile(Dl, 5, axis=1).mean()),
                     P4_TB=float(P4.mean()), G_TB=float(Gc.mean()),
                     R10=dict(D=ci(Dl[:, :10].mean(1)), trang_thai=trang_thai(ci(Dl[:, :10].mean(1)))))
            if P == 2:
                o.update(so_lop_40_dau_TB=float(np.mean(so_lop)), lop_40_dau_max_TB=float(np.mean(lop_max)),
                         histogram_40_dau_TB=(np.mean(np.array(hist, float), axis=(0, 1)).tolist() if hist else None))
            if mr:
                o["R20"] = dict(D=ci(Dl[:, :20].mean(1)), trang_thai=trang_thai(ci(Dl[:, :20].mean(1))))   # giu bang R = 10, 20 canh R cuoi
            out["o"][f"{ds}_{T}_P{P}"] = o
        r0 = [[h for h in p["hoan_vi"] if h["r"] == 0][0] for p in ps]       # co che o thu tu tep
        out["o"][f"{ds}_{T}_tep"] = dict(G_TB=float(np.mean([h["G"] for h in r0])))
    tt = {ds: out["o"][f"{ds}_1.0_P1"]["trang_thai"] for ds in ("benchmark", "beta")}
    cau = {}
    if all(v == "duong" for v in tt.values()):
        cau["chung"] = "tren Benchmark va BETA, E4R-b con loi ich trung binh so voi KTN-R duoi phat lai hoan vi trong khoi (kem Q va CI); khong viet 'khong phu thuoc thu tu'"
    for ds in ("benchmark", "beta"):
        o_ = out["o"][f"{ds}_1.0_P1"]
        d_ = (f"{ds}: D {o_['trang_thai']}" if o_["trang_thai"] == "duong"
              else f"{ds}: D {o_['trang_thai']}; loi ich tren bo nay chi chung minh duoi thu tu tep, kem D [CI] duoi P1")
        q_ = ("duoc noi 'loi ich khong giam qua 0,5 diem so voi thu tu tep'" if o_["Q_khong_giam_qua_0_5"] is True
              else ("Q khong tinh duoc" if o_["Q_khong_giam_qua_0_5"] == "khong tinh duoc" else "chua chung minh khong giam qua 0,5 diem"))
        cau[ds] = d_ + " | Q: " + q_                                      # D va Q xet doc lap
    out["H8"] = dict(trang_thai_1_0=tt, chung_ca_hai_bo=all(v == "duong" for v in tt.values()), cau=cau,
                     tam_thoi=("TAM THOI: luat tang R thoa o R = 20, cho dot r 21..40" if (tang20 is True and not mr) else (tang20 if isinstance(tang20, str) else None)))
    import os
    assert not os.path.exists(a[4]), "tep ra da co: khong ghi de"
    with open(a[4], "x", encoding="utf-8") as fo:
        json.dump(out, fo, indent=1, ensure_ascii=False)
    f = lambda c: "khong tinh duoc" if c["thap"] is None else "%+.2f [%+.2f; %+.2f]" % (c["tb"], c["thap"], c["cao"])
    for key, o in out["o"].items():
        if key.endswith("tep"):
            continue
        print(key, "| D", f(o["D"]), o["trang_thai"], "| Q", f(o["Q"]), "| tep", f(o["thu_tu_tep"]), "| MCSE/SE %.3f" % (o["MCSE_tren_SE"] or float("nan")))
    print("tang R (tinh o R = 20):", tang20, "| R dung:", R, "| H8:", out["H8"])


if __name__ == "__main__":
    main()
