#!/usr/bin/env python
"""Huan luyen LOSO mang nguon W0^EA cho DH-TT.
Chep tu global_dnn_huanluyen.py (cung nap du lieu, cung seed, cung ham huan luyen nhu global_dnn_thichnghi.huan_luyen) va them:
  --ea co   : R_train = TB cov dai 1 cua MOI luot cua cac nguoi NGUON (pt. 6-8), nhan R_train^{-1/2} vao ca 3 dai (pt. 11).
  --tau : entropy cheo tren model(x) / tau (pt. 4; tau = 2). tau = 1 va --ea khong phai tai tao dung W0 (cong hoi quy).
Fold f: hoc tren moi nguoi tru nguoi f; luu foldNNN_trongso.pt va foldNNN_Rtrain.npy; KHONG cham nguoi f (khong in do chinh xac).
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B                  # noqa: E402
import global_dnn as GD              # noqa: E402
import du_lieu_eld as DE             # noqa: E402
import du_lieu_nak as DN             # noqa: E402
import du_lieu_dong as DG            # noqa: E402
import dh_tt_loi as DH               # noqa: E402

SEED = 20261107
SO_LUOT = {**DE.SO_LUOT, **DN.SO_LUOT, **DG.SO_LUOT}
SO_LOP = {**DE.SO_LOP, **DN.SO_LOP, **DG.SO_LOP}
SO_KENH = {"benchmark": 9, "beta": 9, **DG.SO_KENH}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def huan_luyen(Xtr, ytr, C, ns, a, dev, seed, tau):
    """Ban sao global_dnn_thichnghi.huan_luyen; khac duy nhat: loss tren model(x) / tau."""
    model = GD.GlobalDNN(C, ns, nch=a.nch, p_cuoi=a.dropout_cuoi).to(dev)
    w = [p for n, p in model.named_parameters() if n.endswith("weight")]
    b = [p for n, p in model.named_parameters() if n.endswith("bias")]
    opt = torch.optim.Adam([{"params": w, "weight_decay": 1e-3}, {"params": b, "weight_decay": 0.0}], lr=1e-4)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator(device=dev); g.manual_seed(seed)
    n = len(ytr); nb = n // a.lo
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=g)
        for k in range(nb):
            idx = perm[k * a.lo:(k + 1) * a.lo]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xtr[idx]) / tau, ytr[idx]).backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "dong"])
    ap.add_argument("--T", type=float, required=True, choices=[0.6, 1.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ea", required=True, choices=["co", "khong"])
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--folds", default="all", help="'all', 'i::k' hoac 'a-b' (dem tu 1)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--nch", type=int, default=22)
    ap.add_argument("--dropout-cuoi", type=float, default=0.95)
    ap.add_argument("--cho-phep-cpu", action="store_true")
    a = ap.parse_args()
    a.loc = "causal"
    if a.tau <= 0:
        sys.exit("tau phai duong")
    DE.dang_ky(B, GD, a.ds); DN.dang_ky(B, GD, a.ds); DG.dang_ky(B, GD, a.ds)
    for _ds in B.DS:                  # nhu b1gop.py: maychu4 doc ban sao da doi chieu md5 qua B1_ROOT_BENCHMARK, B1_ROOT_BETA
        B.DS[_ds]["root"] = os.environ.get("B1_ROOT_" + _ds.upper(), B.DS[_ds]["root"])
    assert GD.NT == SO_LOP[a.ds]
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    c = B.DS[a.ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    if a.folds == "all":
        folds = list(range(1, nsub + 1))
    elif "::" in a.folds:
        i0, k = (int(v) for v in a.folds.split("::")); folds = list(range(i0, nsub + 1, k))
    else:
        i0, i1 = (int(v) for v in a.folds.split("-")); folds = list(range(i0, i1 + 1))
    assert folds and all(1 <= f <= nsub for f in folds)
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(seed=SEED, ds=a.ds, loc=a.loc, T=a.T, epochs=a.epochs, lo=a.lo, nch=a.nch, dropout_cuoi=a.dropout_cuoi, subjects=0,
               ea=a.ea, tau=a.tau, script_sha256=sha(os.path.abspath(__file__)),
               dh_tt_loi_sha256=sha(DH.__file__), global_dnn_sha256=sha(GD.__file__), b1_diem_sha256=sha(B.__file__),
               du_lieu_dong_sha256=(sha(DG.__file__) if a.ds == "dong" else None), kep=DH.KEP, torch=torch.__version__,
               root_du_lieu=B.DS[a.ds]["root"])
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
        if X.shape[0] != nl or X.shape[1] != SO_KENH[a.ds] or not np.isfinite(X).all() or not np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, nl // GD.NT)):
            sys.exit(f"du lieu nguoi {s} sai cau truc")
        D["X"].append(GD.tien_xu_ly(X, a.ds, a.loc, ns, fs)); D["y"].append(y)
    print(f"nap {nsub} nguoi ({a.ds}), T = {a.T}, ea = {a.ea}, tau = {a.tau}, thiet bi {dev}", flush=True)
    for f in folds:
        i = f - 1
        fn = os.path.join(a.out, f"fold{f:03d}.json")
        if os.path.exists(fn):
            continue
        for hv in (f"fold{f:03d}_trongso.pt", f"fold{f:03d}_Rtrain.npy"):     # fold do dang: khong ghi de hien vat cu
            if os.path.exists(os.path.join(a.out, hv)):
                sys.exit(f"{hv} da co ma chua co fold{f:03d}.json: kiem tay, khong ghi de")
        seed = SEED + 1000 * f + int(round(a.T * 10))
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        tr = [s for s in range(nsub) if s != i]
        assert i not in tr and len(tr) == nsub - 1                              # chong ro ri: chi nguoi nguon
        Xnp = np.concatenate([D["X"][s] for s in tr])
        if a.ea == "co":
            Rt = DH.r_nguon(Xnp); P = DH.nghich_can(Rt)
            assert np.isfinite(P).all()
            Xnp = DH.can_chinh(Xnp, P)
            np.save(os.path.join(a.out, f"fold{f:03d}_Rtrain.npy"), Rt)
        Xtr = torch.tensor(Xnp, device=dev)
        ytr = torch.tensor(np.concatenate([D["y"][s] for s in tr]), device=dev, dtype=torch.long)
        del Xnp
        t0 = time.time()
        model = huan_luyen(Xtr, ytr, Xtr.shape[2], Xtr.shape[3], a, dev, seed, a.tau)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values()), f"trong so khong huu han o fold {f}"
        torch.save(model.state_dict(), os.path.join(a.out, f"fold{f:03d}_trongso.pt"))
        del Xtr, ytr
        js = dict(fold=f, seed=seed, nguoi_bi_loai=f, so_nguoi_hoc=len(tr), giay_huan_luyen=time.time() - t0)
        json.dump(js, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
        print(f"fold {f}: xong {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
