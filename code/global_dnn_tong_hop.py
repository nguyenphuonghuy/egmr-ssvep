#!/usr/bin/env python
"""Tong hop mot luot Global DNN (global_dnn.py): kiem du fold, cung config, 240 luot moi nguoi; ghi CSV do chinh xac tung nguoi (%).
--cong lo,hi: cong tai hien (luot A filtfilt 1,0 s phai trong [67,61; 73,61] %); ngoai cong thi thoat ma 3 (job dung truoc luot B).
"""
import argparse, csv, glob, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--nsub", type=int, default=35)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cong", default="")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.res, "config.json")))
    rows = []
    for f in range(1, a.nsub + 1):
        fn = os.path.join(a.res, f"fold{f:03d}.json")
        if not os.path.exists(fn):
            sys.exit(f"thieu {fn}")
        d = json.load(open(fn))
        if d["fold"] != f or d["n_test"] != 240 or d["epochs"] != cfg["epochs"] or d["loc"] != cfg["loc"] or d["T"] != cfg["T"] \
                or d.get("nch", 120) != cfg.get("nch", 120) or d.get("dropout_cuoi", 0.95) != cfg.get("dropout_cuoi", 0.95) \
                or not np.isfinite(d["acc"]):
            sys.exit(f"fold {f} khong khop config hoac sai so luot")
        rows.append((f, 100 * d["acc"], d["seed"], d["giay"]))
    extra = sorted(glob.glob(os.path.join(a.res, "fold*.json")))
    if len(extra) != a.nsub:
        sys.exit(f"so tep fold {len(extra)} khac {a.nsub}")
    if os.path.exists(a.csv):
        sys.exit(f"{a.csv} da co: khong ghi de")
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["nguoi", "acc_pct", "seed", "giay"]); w.writerows(rows)
    acc = np.array([r[1] for r in rows])
    tb = float(acc.mean())
    tom = dict(res=a.res, loc=cfg["loc"], T=cfg["T"], epochs=cfg["epochs"], n=len(acc), tb_pct=tb, sd_pct=float(acc.std(ddof=1)),
               se_pct=float(acc.std(ddof=1) / np.sqrt(len(acc))), mac=json.load(open(os.path.join(a.res, "fold001.json")))["mac"],
               script_sha256=cfg["script_sha256"], cong=a.cong, nch=cfg.get("nch", 120),
               dropout_cuoi=cfg.get("dropout_cuoi", 0.95))
    json.dump(tom, open(a.csv.replace(".csv", "_tom-tat.json"), "w"), indent=1)
    print(json.dumps(tom))
    if a.cong:
        lo, hi = (float(v) for v in a.cong.split(","))
        if not (lo <= tb <= hi):
            print(f"NGOAI CONG: TB {tb:.2f} % ngoai [{lo}; {hi}]")
            sys.exit(3)
        print(f"QUA CONG: TB {tb:.2f} % trong [{lo}; {hi}]")


if __name__ == "__main__":
    main()
