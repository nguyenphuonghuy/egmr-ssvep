#!/usr/bin/env python
"""Dong goi du lieu cho THI NGHIEM B de chep len ZCU104 (board khong doc duoc HPC).
Chay tren HPC. Moi nguoi: EEG tho 9 kenh (luot, 9, t) theo thu tu khoi roi muc tieu, nhan, tan so, pha (Benchmark). Them thu tu P1 r = 1 cua nguoi 1
(cung ham sinh voi b1gop_hoanvi.py: np.random.default_rng([20261107, 1, 1, 1]), permutation lan luot tung khoi 40 luot).
Nguoi: 1 (B1, B2), 17 va 27 (B3: fold trung vi va lon nhat theo so vong a = 220, da chot o phu luc). Ghi manifest.json: bam cac tep npz va bam trong
so fold 17, 27 lay tu JSON SFDA-TT diem 6, kiem lai so vong a = 220 cua hai fold voi JSON do.
Dung: python zcu104_doi_thu_goi.py <thu muc ra> <sfda_tt_benchmark_T10.json> <thu muc trong so E4 T10>
"""
import hashlib, json, os, sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B  # noqa: E402

for d in B.DS:
    B.DS[d]["root"] = os.environ.get("B1_ROOT_" + d.upper(), B.DS[d]["root"])
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()


def main():
    ra, fj, wdir = sys.argv[1:4]
    os.makedirs(ra, exist_ok=False)
    j = json.load(open(fj))
    assert j["ds"] == "benchmark" and j["che_do"] == "tt" and abs(j["T"] - 1.0) < 1e-9
    tong220 = {p["fold"]: sum(m["so_lan"]) if isinstance(m["so_lan"], list) else None for p in j["tung_nguoi"] for m in p["moc"] if m["a"] == 220}
    v = sorted(tong220.values()); tv = v[len(v) // 2]
    assert tv == 61 and min(f for f, x in tong220.items() if x == tv) == 17 and max(tong220, key=tong220.get) == 27 and tong220[27] == 168, "chon fold khac phu luc"
    mf = dict(sfda_tt_json=os.path.basename(fj), sfda_tt_json_sha256=sha(fj), trongso_sha256={}, npz_sha256={})
    for f in (17, 27):
        w = os.path.join(wdir, f"fold{f:03d}_trongso.pt")
        mf["trongso_sha256"][str(f)] = j["tung_nguoi"][f - 1]["trongso_sha256"]
        assert sha(w) == mf["trongso_sha256"][str(f)], f"trong so fold {f} khac JSON d6"
    c = B.DS["benchmark"]
    ph = np.asarray(loadmat(os.path.join(c["root"], "Freq_Phase.mat"))["phases"]).ravel().astype(float)
    for s in (1, 17, 27):
        X, y, freqs = B.load_subject("benchmark", s, B.CH9)
        assert X.shape[:2] == (240, 9) and np.array_equal(y, np.tile(np.arange(40), 6)) and np.isfinite(X).all()
        them = {}
        if s == 1:
            rng = np.random.default_rng([20261107, 1, 1, 1])
            them["pi_P1_r1"] = np.concatenate([b * 40 + rng.permutation(40) for b in range(6)])
        fn = os.path.join(ra, f"benchmark_S{s}.npz")
        np.savez(fn, X=X, y=y, freqs=np.asarray(freqs, float), phases=ph, **them)
        mf["npz_sha256"][os.path.basename(fn)] = sha(fn)
        print(os.path.basename(fn), mf["npz_sha256"][os.path.basename(fn)], X.shape, X.dtype)
    json.dump(mf, open(os.path.join(ra, "manifest.json"), "w"), indent=1)
    print("manifest", mf["trongso_sha256"])


if __name__ == "__main__":
    main()
