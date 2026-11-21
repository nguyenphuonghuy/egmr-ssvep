#!/usr/bin/env python
"""Bo nap eldBETA cho mang nen va phat lai (2026-11-15). Cach doc GIONG HET b1moi_eld.py (job 18681, da kiem):
data.EEG.Epoch (64 kenh, 1500 mau, 9 muc tieu, 7 khoi), fs 250, 0,5 s truoc kich thich (onset 125), do tre 0,14 s, 9 kenh CH9,
khong loc chan them (da loc chan phan cung, kiem pho o eldbeta_kiem_du_lieu.py). Thu tu luot: khoi roi muc tieu; y = tile(arange(9), 7).
dang_ky(B, GD): them B.DS["eld"], boc B.load_subject de nhan ds "eld", va dat so lop 9 cho Global DNN (GD.NT va gia tri mac dinh nclass).
"""
import os, functools
import numpy as np

ELD = "/gpfs0/users/gv_dnton/datasets/ssvep/eldbeta"
F_TEP = np.array([8.0, 9.5, 11.0, 8.5, 10.0, 11.5, 9.0, 10.5, 12.0])
SO_LOP = {"benchmark": 40, "beta": 40, "eld": 9}
SO_LUOT = {"benchmark": 240, "beta": 160, "eld": 63}


def nap(s, chans):
    from scipy.io import loadmat
    dd = loadmat(os.path.join(ELD, "S%d.mat" % s), squeeze_me=True, struct_as_record=False)["data"]
    E = dd.EEG.Epoch; si = dd.Suppl_info
    assert E.shape == (64, 1500, 9, 7) and np.isfinite(E).all(), (s, E.shape)
    names = [str(r[3]).strip().upper() for r in si.Channel]
    fr = np.asarray(si.Frequency, float).ravel()
    assert np.allclose(fr, F_TEP) and int(si.Srate) == 250
    X = E[[names.index(c) for c in chans]].transpose(3, 2, 0, 1).reshape(63, len(chans), 1500).astype(np.float64)   # khoi-chinh, muc tieu 0..8
    return X, np.tile(np.arange(9), 7), fr


def dang_ky_B(B):
    """Chi dang ky bo nap (khong dung torch): them B.DS["eld"], boc B.load_subject."""
    if "eld" in B.DS:
        return
    B.DS["eld"] = dict(root=ELD, n=100, fs=250, onset=125, lat=0.14)
    goc = B.load_subject

    def ls(d, s, chans):
        return nap(s, chans) if d == "eld" else goc(d, s, chans)
    B.load_subject = ls


def dang_ky(B, GD, ds):
    """Goi MOT lan truoc khi dung B.load_subject / GD.GlobalDNN. ds khac 'eld' thi khong doi gi."""
    if ds != "eld":
        return
    B.DS["eld"] = dict(root=ELD, n=100, fs=250, onset=125, lat=0.14)
    goc = B.load_subject

    def ls(d, s, chans):
        return nap(s, chans) if d == "eld" else goc(d, s, chans)
    B.load_subject = ls
    GD.NT = 9                                                    # dem_mac doc GD.NT luc goi
    GD.GlobalDNN = functools.partial(GD.GlobalDNN, nclass=9)     # FC 9 lop (huan_luyen va b1moi_lai deu goi GD.GlobalDNN luc chay)
