#!/usr/bin/env python
"""BAN CHEP cua global_dnn_huanluyen.py cho nhanh b1gop (2026-11-15): chi doi bo nap eld -> lee2019 (du_lieu_lee.py), MOI phan khac giu nguyen.
Khong sua tep goc vi nhanh KTN dang dung ma bam cua no.
Huan luyen LOSO Global DNN thu nho (nch 22, dropout cuoi 0,95, loc nhan qua) cho mot bo du lieu (benchmark | beta | lee), CHI luu trong so va logit tinh.
Dung lai NGUYEN ham huan_luyen cua global_dnn_thichnghi.py (lo 2) de mang nen cua BETA duoc huan luyen dung nhu mang nen cua Benchmark.
Fold i: hoc tren moi nguoi tru nguoi i; luu foldNNN_trongso.pt TRUOC khi cham nguoi i; logit tinh cua nguoi i (nhan chi de cham).
config.json cung khoa voi lo 2 (nch, loc, T, subjects, script_sha256) de b1moi_lai.py doc duoc.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B                  # noqa: E402
import global_dnn as GD              # noqa: E402
import global_dnn_thichnghi as GT    # noqa: E402
import du_lieu_lee as DE             # noqa: E402

SEED = 20261107
SO_LUOT = DE.SO_LUOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "lee"])
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--nch", type=int, default=22)
    ap.add_argument("--dropout-cuoi", type=float, default=0.95)
    ap.add_argument("--cho-phep-cpu", action="store_true")
    a = ap.parse_args()
    a.loc = "causal"
    DE.dang_ky(B, GD, a.ds)
    assert GD.NT == DE.SO_LOP[a.ds]
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    c = B.DS[a.ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    if a.folds == "all":
        folds = list(range(1, nsub + 1))
    else:
        i0, k = (int(v) for v in a.folds.split("::")); folds = list(range(i0, nsub + 1, k))
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(seed=SEED, ds=a.ds, loc=a.loc, T=a.T, epochs=a.epochs, lo=a.lo, nch=a.nch, dropout_cuoi=a.dropout_cuoi, subjects=0,
               script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
               thichnghi_sha256=hashlib.sha256(open(GT.__file__, "rb").read()).hexdigest(),
               global_dnn_sha256=hashlib.sha256(open(GD.__file__, "rb").read()).hexdigest(),
               b1_diem_sha256=hashlib.sha256(open(B.__file__, "rb").read()).hexdigest(),
               du_lieu_lee_sha256=hashlib.sha256(open(DE.__file__, "rb").read()).hexdigest(),
               lee_manifest_sha256=(DE.ma_manifest() if a.ds == "lee" else None))
    cfgf = os.path.join(a.out, "config.json")
    if os.path.exists(cfgf):
        if json.load(open(cfgf)) != cfg:
            sys.exit("config khac lan chay truoc: khong tron ket qua")
    else:
        tmp = f"{cfgf}.{os.getpid()}.tmp"; json.dump(cfg, open(tmp, "w"), indent=1); os.replace(tmp, cfgf)
    D = {"X": [], "y": []}
    nl = SO_LUOT[a.ds]
    for s in range(1, nsub + 1):
        X, y, _ = B.load_subject(a.ds, s, B.CH9)
        if X.shape[0] != nl or X.shape[1] != 9 or not np.isfinite(X).all() or not np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, nl // GD.NT)):
            sys.exit(f"du lieu nguoi {s} sai cau truc")
        D["X"].append(GD.tien_xu_ly(X, a.ds, a.loc, ns, fs)); D["y"].append(y)
    print(f"nap {nsub} nguoi ({a.ds}), T = {a.T}, thiet bi {dev}", flush=True)
    for f in folds:
        i = f - 1
        fn = os.path.join(a.out, f"fold{f:03d}.json")
        if os.path.exists(fn):
            continue
        seed = SEED + 1000 * f + int(round(a.T * 10))
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        tr = [s for s in range(nsub) if s != i]
        assert i not in tr and len(tr) == nsub - 1
        Xtr = torch.tensor(np.concatenate([D["X"][s] for s in tr]), device=dev)
        ytr = torch.tensor(np.concatenate([D["y"][s] for s in tr]), device=dev, dtype=torch.long)
        t0 = time.time()
        model = GT.huan_luyen(Xtr, ytr, Xtr.shape[2], Xtr.shape[3], a, dev, seed)
        torch.save(model.state_dict(), os.path.join(a.out, f"fold{f:03d}_trongso.pt"))     # luu TRUOC khi cham nguoi i
        del Xtr, ytr
        model.eval()
        with torch.no_grad():
            lg = model(torch.tensor(D["X"][i], device=dev)).cpu().numpy()
        y = D["y"][i]
        np.savez_compressed(os.path.join(a.out, f"fold{f:03d}_logit-tinh.npz"), logit=lg.astype(np.float32), y=y)
        js = dict(fold=f, seed=seed, nguoi_bi_loai=f, so_nguoi_hoc=len(tr), tinh_tat_ca=float((lg.argmax(1) == y).mean()),
                  tinh_sau_40=float((lg[40:].argmax(1) == y[40:]).mean()), giay_huan_luyen=time.time() - t0)
        json.dump(js, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
        print(f"fold {f}: xong {time.time()-t0:.0f} s", flush=True)                         # khong in do chinh xac (bo xac nhan)


if __name__ == "__main__":
    main()
