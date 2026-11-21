#!/usr/bin/env python
"""Bo nap Nakanishi 2015 (12JFPM; github.com/mnakanishi/12JFPM_SSVEP) cho buoc 2 KTN-R.
Tep sN.mat, bien 'eeg' (12 muc tieu x 8 kenh x 1114 mau x 15 khoi), fs 256 Hz, du lieu tho da giam mau, CHUA loc (Claude do tren s1, 2026-11-16).
Onset o mau thu 39 (chi so 0-based 38) theo README; do tre thi giac 0,135 s theo bai goc (round(0,135 x 256) = 35 mau): cua so bat dau chi so 73.
Ten 8 kenh khong ghi trong van ban bai goc; dung ca 8 kenh theo thu tu tep (giong nhau o moi nguoi). Khong loc chan (nhu eldBETA va CCA-Net).
Thu tu luot: khoi roi muc tieu (thu tu DUNG LAI, bai goc xep muc tieu ngau nhien trong khoi, khong co metadata); y = tile(arange(12), 15).
Tan so theo thu tu muc tieu trong tep (README): 9,25 11,25 13,25 9,75 11,75 13,75 10,25 12,25 14,25 10,75 12,75 14,75 Hz.
dang_ky(B, GD, ds): ds == "nak" thi them B.DS["nak"], boc B.load_subject (bo qua danh sach ten kenh), dat 12 lop cho Global DNN.
"""
import os, functools
import numpy as np

NAK = "/gpfs0/users/gv_dnton/datasets/ssvep/nakanishi2015"
F_NAK = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
SO_LOP = {"nak": 12}
SO_LUOT = {"nak": 180}
SO_KENH = {"nak": 8}
CAU_HINH = dict(root=NAK, n=10, fs=256, onset=38, lat=0.135)


def nap(s, chans=None):
    from scipy.io import loadmat
    E = loadmat(os.path.join(NAK, "s%d.mat" % s))["eeg"]
    assert E.shape == (12, 8, 1114, 15) and np.isfinite(E).all(), (s, E.shape)
    X = E.transpose(3, 0, 1, 2).reshape(180, 8, 1114).astype(np.float64)     # khoi-chinh, muc tieu 0..11
    return X, np.tile(np.arange(12), 15), F_NAK.copy()


def dang_ky(B, GD, ds):
    """Goi MOT lan truoc khi dung B.load_subject / GD.GlobalDNN. ds khac 'nak' thi khong doi gi."""
    if ds != "nak" or "nak" in B.DS:
        return
    B.DS["nak"] = dict(CAU_HINH)
    goc = B.load_subject

    def ls(d, s, chans):
        return nap(s) if d == "nak" else goc(d, s, chans)
    B.load_subject = ls
    if GD is not None:
        GD.NT = 12
        GD.GlobalDNN = functools.partial(GD.GlobalDNN, nclass=12)
