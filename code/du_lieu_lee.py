#!/usr/bin/env python
"""Bo nap lee2019 (ban tien xu ly gon tu lee_chuyen_derived.py) cho mang nen va phat lai nhanh b1gop (2026-11-15).
Cach doc giong b1moi_lee.py (cau hinh khoa 2026-11-14): 9 kenh Pz PO9 PO3 POz PO4 PO10 O1 Oz O2 (thay PO5, PO6; giu C = 9), fs 250, onset 1 (mau 0
lam moc), lat 0,18 s (0,14 s + 0,04 s tre nhom FIR ha mau). 54 nguoi, 200 luot, 4 lop (50 moi lop). Thu tu luot = thu tu trong tep (thu tu ghi that).
dang_ky(B, GD, ds): ds == "lee" thi them B.DS["lee"], boc B.load_subject, dat so lop 4 cho Global DNN; ds khac thi khong doi gi.
Thu muc goc lay tu bien moi truong B1_ROOT_LEE (mac dinh: thu muc derived cua project tren maychu4).
"""
import hashlib, json, os, functools
import numpy as np

LEE = os.environ.get("B1_ROOT_LEE", "/gpfs0/users/gv_mtthai/projects/qldanc_ton_b1gop/data/lee2019_derived_v1")
F_TEP = np.array([12.0, 8.57, 6.67, 5.45])
SO_LOP = {"benchmark": 40, "beta": 40, "lee": 4}
SO_LUOT = {"benchmark": 240, "beta": 160, "lee": 200}
MA_CHUYEN = "e59e0e08ba8c204c5c8effd1fc1a98293ec400f6b6ecbbaf2b8c7d4dc54971fa"   # sha256 lee_chuyen_derived.py


def _manifest():
    return json.load(open(os.path.join(LEE, "manifest.json")))


def ma_manifest():
    """sha256 cua manifest.json cua ban derived (ghi vao config trong so va doi chieu khi phat lai)."""
    return hashlib.sha256(open(os.path.join(LEE, "manifest.json"), "rb").read()).hexdigest()


def nap(s, chans, CH9):
    assert list(chans) == list(CH9), "lee2019 chi doc dung bo 9 kenh co dinh (PO9, PO10 thay PO5, PO6)"
    fn = os.path.join(LEE, "lee2019_s%02d.npz" % s)
    m = _manifest()
    assert m["kenh"] == ["PZ", "PO9", "PO3", "POZ", "PO4", "PO10", "O1", "OZ", "O2"] and m["fs"] == 250 and m["onset"] == 1 and m["lat"] == 0.18 and m["chan60"] is False
    assert m["ma_sha256"] == MA_CHUYEN, "ban derived khong do lee_chuyen_derived.py da kiem tao ra"
    assert hashlib.md5(open(fn, "rb").read()).hexdigest() == m["md5"][os.path.basename(fn)], "md5 tep derived khac manifest: " + fn
    d = np.load(fn)
    X, y = d["X"], d["y"].astype(int)
    assert bool(d["t_tang"]), "luot khong tang theo thoi gian trong tung phan: trai voi gia dinh thu tu ghi that"
    assert X.shape == (200, 9, 1001) and np.isfinite(X).all() and np.array_equal(np.bincount(y, minlength=4), [50, 50, 50, 50]), s
    return X, y, F_TEP


def dang_ky(B, GD, ds):
    if ds != "lee":
        return
    B.DS["lee"] = dict(root=LEE, n=54, fs=250, onset=1, lat=0.18)
    goc = B.load_subject

    def ls(d, s, chans):
        return nap(s, chans, B.CH9) if d == "lee" else goc(d, s, chans)
    B.load_subject = ls
    GD.NT = 4
    GD.GlobalDNN = functools.partial(GD.GlobalDNN, nclass=4)
