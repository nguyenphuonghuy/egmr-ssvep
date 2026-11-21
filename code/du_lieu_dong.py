#!/usr/bin/env python
"""Bo nap Dong 2023 (Brain Science Advances 9(4):297-309, DOI 10.26599/BSA.2023.9050020; Zenodo 18847318) cho xac nhan KTN-R
Tep SN.mat, bien 'eegdata' (8 kenh x 1250 mau x 40 muc tieu x 4 khoi), fs 250 Hz, epoch co 0,5 s truoc kich thich (onset chi so 125),
da loc chan 50 Hz, chua loc thong dai. Kenh theo bai: POz, PO3, PO4, PO7, PO8, Oz, O1, O2 (Oz chi so 5). Do tre thi giac 0,16 s (bai uoc 159,49 ms).
Tan so muc tieu k: 8 + 0,2 k Hz, pha (k mod 4) x 0,5 pi (hinh 1b cua bai; kiem anh xa bang dong_kiem_du_lieu.py truoc khi chay).
Thu tu luot: khoi roi muc tieu (dung lai; thu tu that ngau nhien, khong co metadata); y = tile(arange(40), 4).
dang_ky(B, GD, ds): ds == "dong" thi them B.DS["dong"], boc B.load_subject (bo qua danh sach ten kenh), dat 40 lop.
"""
import os, functools
import numpy as np

DONG = "/gpfs0/users/gv_dnton/datasets/ssvep/dong2023"
F_DONG = 8.0 + 0.2 * np.arange(40)
PHA_DONG_PI = 0.5 * (np.arange(40) % 4)
OZ = 5
SO_LOP = {"dong": 40}
SO_LUOT = {"dong": 160}
SO_KENH = {"dong": 8}
CAU_HINH = dict(root=DONG, n=59, fs=250, onset=125, lat=0.16)


def nap(s, chans=None):
    from scipy.io import loadmat
    E = loadmat(os.path.join(DONG, "S%d.mat" % s))["eegdata"]
    assert E.shape == (8, 1250, 40, 4) and np.isfinite(E).all(), (s, E.shape)
    X = E.transpose(3, 2, 0, 1).reshape(160, 8, 1250).astype(np.float64)     # khoi-chinh, muc tieu 0..39
    return X, np.tile(np.arange(40), 4), F_DONG.copy()


def dang_ky(B, GD, ds):
    """Goi MOT lan truoc khi dung B.load_subject / GD.GlobalDNN. ds khac 'dong' thi khong doi gi."""
    if ds != "dong" or "dong" in B.DS:
        return
    B.DS["dong"] = dict(CAU_HINH)
    goc = B.load_subject

    def ls(d, s, chans):
        return nap(s) if d == "dong" else goc(d, s, chans)
    B.load_subject = ls
    if GD is not None:
        GD.NT = 40
