#!/usr/bin/env python
"""Phan tich doi chung entropy.
Doc 6 JSON b2e4r_entropy (3 bo x 2 cua so), kiem du fold, tinh CI, ghi JSON.
CI: BCa 95 % qua nguoi, 10 000 lan, seed 20261107 (scipy.stats.bootstrap, paired qua chenh lech tung nguoi); suy bien thi "khong tinh duoc".
Dung: python b2e4r_entropy_phan_tich.py <thu muc JSON> <json ra>
"""
import json, os, sys, hashlib
import numpy as np
from scipy import stats

B2_SHA = {"eb6f7225ee62093865f13c9cf5caa0ebe4dc00e092f44e3705be69abf6962dad", "e56f4877ff9e0b36b430e441ce36ce2d37abf9d78d7f46f6bd26d8daffc918b9"}
med = lambda v: float(np.median(v)) if len(v) else None
tb_ = lambda v: float(np.mean(v)) if len(v) else None
BO = ("benchmark", "beta", "dong"); TEN = {"benchmark": "Benchmark", "beta": "BETA", "dong": "Dong 2023"}
SO_NGUOI = {"benchmark": 35, "beta": 70, "dong": 59}


def ci(d):
    """Cung cach voi b2e4r_phan_tich.ci: nhan 100 truoc, BCa 10 000 lan, seed 20261107; suy bien -> None."""
    d = 100 * np.asarray(d, float)
    if not np.isfinite(d).all() or np.allclose(d, d[0]):
        return None
    try:
        r = stats.bootstrap((d,), np.mean, n_resamples=10000, method="BCa", random_state=np.random.default_rng(20261107)).confidence_interval
    except Exception:
        return None
    lo, hi = float(r.low), float(r.high)
    return None if not (np.isfinite(lo) and np.isfinite(hi)) else (lo, hi)


def trang_thai(c):
    if c is None:
        return "khong tinh duoc"
    return "hon" if c[0] > 0 else ("kem" if c[1] < 0 else "chua phan biet")


def main():
    D, ra = sys.argv[1], sys.argv[2]
    assert not os.path.exists(ra), "tep ra da co"
    out = dict(script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(), o={}, nguon={})
    for b in BO:
        for T, tt in ((1.0, "10"), (0.6, "06")):
            fn = [f for f in os.listdir(D) if f.startswith(f"{b}_b2e4r-entropy_T{tt}_") and f.endswith(".json")]
            assert len(fn) == 1, (b, tt, fn)
            p = os.path.join(D, fn[0]); js = json.load(open(p))
            out["nguon"][f"{b}_{tt}"] = dict(tep=fn[0], sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(), script_sha256=js["script_sha256"],
                                             b2e4r_sha256=js["b2e4r_sha256"], json_b2e4r=js["json_b2e4r"], json_b2e4r_sha256=js["json_b2e4r_sha256"])
            assert js["du_fold"] and abs(js["T"] - T) < 1e-9 and js["ds"] == b
            assert js["n0"] == 40 and js["alpha"] == 0.5 and js["seed"] == 20261107 and js["nhanh"] == ["b", "Hs", "Hz", "Hl"]
            assert js["b2e4r_sha256"] in B2_SHA, "b2e4r.py khong phai ban da sinh G0/G1/X1"
            P = js["tung_nguoi"]; assert [x["fold"] for x in P] == list(range(1, SO_NGUOI[b] + 1))
            A = {k: np.array([x[k] for x in P]) for k in ("KTNR", "E4Ra", "E4Rb", "Hs", "Hz", "Hl", "P4_b", "P4_Hs")}
            o = {}
            for ma, d in (("H9", A["E4Rb"] - A["Hs"]), ("H9a", A["Hs"] - A["KTNR"]), ("H9b", A["E4Ra"] - A["Hz"]), ("H9e", A["Hz"] - A["Hl"])):
                c = ci(d); o[ma] = dict(tb=100 * float(d.mean()), ci=c, trang_thai=trang_thai(c))
            n_luot = int(sum(x["n_cham"] for x in P))
            o["H9c"] = dict(tb=float(A["P4_Hs"].mean() / A["P4_b"].mean()) if A["P4_b"].mean() > 0 else None)
            rh = [x["them_Hs"]["rho_tb"] for x in P if x["them_Hs"].get("rho_tb") is not None]
            o["H9d"] = dict(tb=(float(np.median(rh)) if rh else None), so_nguoi_bo=len(P) - len(rh))
            o["TB"] = {k: 100 * float(A[k].mean()) for k in ("KTNR", "E4Ra", "E4Rb", "Hs", "Hz", "Hl")}
            o["P4_Hs_tb"] = float(A["P4_Hs"].mean()); o["P4_b_tb"] = float(A["P4_b"].mean())
            if o["H9a"]["trang_thai"] == "hon" and o["P4_Hs_tb"] < 0.02:          # nhu b2e4r_phan_tich
                o["H9a"]["trang_thai"] = "khong kiem duoc (P4 < 0,02)"
            bo = {n_: int(sum(x["them_" + n_]["bo_khong_huu_han"] for x in P)) for n_ in ("b", "Hs", "Hz", "Hl")}
            o["ty_le_bo"] = {n_: bo[n_] / n_luot for n_ in bo}                     # ti le bo theo nhanh
            o["can_xem_loi_so_hoc"] = bool(any(bo.values()))                       # co: khong dung cau uu the truoc khi xem anh huong
            o["chan_doan"] = dict(
                rho_tho_trung_vi=med([x["them_Hs"]["rho_tho_tb"] for x in P if x["them_Hs"].get("rho_tho_tb") is not None]),
                L1_lam_b_Hs_tb=float(np.mean([x["L1_lam_b_Hs_tb"] for x in P])),
                bo_khong_huu_han=bo,
                r_tb={n_: tb_([x["them_" + n_]["r_tb"] for x in P if x["them_" + n_]["r_tb"] is not None]) for n_ in ("b", "Hs", "Hz", "Hl")},
                buoc_tb={n_: tb_([x["them_" + n_]["buoc_tb"] for x in P if x["them_" + n_]["buoc_tb"] is not None]) for n_ in ("b", "Hs", "Hz", "Hl")},
                G_trung_vi={n_: med([x["them_" + n_]["G"] for x in P]) for n_ in ("b", "Hs", "Hz", "Hl")},
                bo_G0={n_: int(sum(x["them_" + n_]["bo_G0"] for x in P)) for n_ in ("b", "Hs", "Hz", "Hl")},
                tv_tb={n_: float(np.mean([x["them_" + n_]["tv"] for x in P])) for n_ in ("b", "Hs", "Hz", "Hl")},
                Hl_H_trung_vi=float(np.median([x["them_Hl"]["H_trung_vi"] for x in P])),
                Hl_bien_trung_vi=float(np.median([x["them_Hl"]["bien_trung_vi"] for x in P])))
            out["o"][f"{b}_{tt}"] = o
    with open(ra, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, o in out["o"].items():
        print(k, {m: (round(o[m]["tb"], 3) if o[m].get("tb") is not None else None, o[m].get("trang_thai")) for m in ("H9", "H9a", "H9b", "H9e", "H9c", "H9d")})


if __name__ == "__main__":
    main()
