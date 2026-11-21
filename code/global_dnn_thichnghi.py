#!/usr/bin/env python
"""LO 2: thich nghi KHONG NHAN cho Global DNN thu nho, cung giao thuc voi lo 1
(b1v7_lo_caitien.py): khoi chan cua nguoi thu = luong tich luy khong nhan, khoi le = phan giu lai de cham.

Moi fold i (LOSO Benchmark): huan luyen Global DNN thu nho (nch 22, dropout cuoi 0,95, loc nhan qua; cong thuc goc nhu
global_dnn.py) tren 34 nguoi nguon, luu trong so. Sau do voi n luot dau cua luong (P hoan vi), ba cach thich nghi, sieu tham so
CO DINH (khong chinh): Adam lr 1e-4, L2 1e-3 tren trong so, 5 vong x 10 buoc ca lo, nhan gia lam lai dau moi vong:
  fc_pl  : chi lop FC, entropy cheo voi nhan gia
  all_pl : toan mang, entropy cheo voi nhan gia
  tent   : toan mang, cuc tieu entropy du doan (khong nhan gia)
  *_e    : nhu tren nhung TAT dropout khi thich nghi (dropout 0,95 co the lam thiet doi thu)
Cham tren phan giu lai (model.eval). Luu logit tinh cua phan giu lai (npz) de gop diem voi bo doi sanh mau sau.
Nhan that cua nguoi thu chi dung de cham diem va bao ti le nhan gia dung.
"""
import argparse, copy, json, os, sys, time, hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402

SEED = 20261107
NS_LIST = [20, 40, 120]
P_HV = 3
VONG, BUOC = 5, 10
MODES = ["fc_pl", "all_pl", "tent", "fc_pl_e", "all_pl_e", "tent_e"]   # hau to _e: tat dropout khi thich nghi


def huan_luyen(Xtr, ytr, C, ns, a, dev, seed):
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
            lossf(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
    return model


def thich_nghi(model0, Xs, mode):
    m = copy.deepcopy(model0)
    for p in m.parameters():
        p.requires_grad_(False)
    tat_do = mode.endswith("_e"); mode = mode[:-2] if tat_do else mode
    mods = [m.fc] if mode == "fc_pl" else [m]
    ps = [(n, p) for mod in mods for n, p in mod.named_parameters()]
    for _, p in ps:
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [p for n, p in ps if n.endswith("weight")], "weight_decay": 1e-3},
                            {"params": [p for n, p in ps if n.endswith("bias")], "weight_decay": 0.0}], lr=1e-4)
    lossf = nn.CrossEntropyLoss()
    for _ in range(VONG):
        m.eval()
        with torch.no_grad():
            lab = m(Xs).argmax(1)
        if not tat_do:
            m.train()
        for _ in range(BUOC):
            opt.zero_grad(set_to_none=True)
            out = m(Xs)
            if mode == "tent":
                lp = torch.log_softmax(out, 1); loss = -(lp.exp() * lp).sum(1).mean()
            else:
                loss = lossf(out, lab)
            loss.backward(); opt.step()
    m.eval()
    return m


def chay_fold(i, D, a, dev):
    fn = os.path.join(a.out, f"fold{i+1:03d}.json")
    if os.path.exists(fn):
        return
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tr = [s for s in range(len(D["X"])) if s != i]
    assert i not in tr
    Xtr = torch.tensor(np.concatenate([D["X"][s] for s in tr]), device=dev)
    ytr = torch.tensor(np.concatenate([D["y"][s] for s in tr]), device=dev, dtype=torch.long)
    C, ns = Xtr.shape[2], Xtr.shape[3]
    t0 = time.time()
    model = huan_luyen(Xtr, ytr, C, ns, a, dev, seed)
    t_hl = time.time() - t0
    torch.save(model.state_dict(), os.path.join(a.out, f"fold{i+1:03d}_trongso.pt"))
    del Xtr, ytr
    y = D["y"][i]; nblk = len(y) // GD.NT
    blk = np.repeat(np.arange(nblk), GD.NT)
    i_st = np.where(blk % 2 == 0)[0]; i_ho = np.where(blk % 2 == 1)[0]
    Xu = torch.tensor(D["X"][i], device=dev)
    yh = y[i_ho]
    model.eval()
    with torch.no_grad():
        lg = model(Xu).cpu().numpy()
    np.savez_compressed(os.path.join(a.out, f"fold{i+1:03d}_logit-tinh.npz"), logit=lg.astype(np.float32), y=y, i_ho=i_ho)
    res = {"tinh_giu_lai": float((lg[i_ho].argmax(1) == yh).mean()), "tinh_tat_ca": float((lg.argmax(1) == y).mean())}
    rng = np.random.default_rng(seed)
    ns_list = [n for n in NS_LIST if n <= len(i_st)]
    acc = {m: np.zeros((P_HV, len(ns_list))) for m in MODES}
    pd = np.zeros((P_HV, len(ns_list)))
    for p in range(P_HV):
        od = rng.permutation(i_st)
        for j, n in enumerate(ns_list):
            idx = od[:n]; Xs = Xu[torch.tensor(idx, device=dev)]
            pd[p, j] = (lg[idx].argmax(1) == y[idx]).mean()
            for m in MODES:
                torch.manual_seed(seed + 17 * p + j)
                ma = thich_nghi(model, Xs, m)
                with torch.no_grad():
                    acc[m][p, j] = (ma(Xu[torch.tensor(i_ho, device=dev)]).argmax(1).cpu().numpy() == yh).mean()
    for m in MODES:
        res[m] = acc[m].mean(0).tolist()
    res["nhan_gia_dung_ban_dau"] = pd.mean(0).tolist()
    js = dict(fold=i + 1, seed=seed, T=a.T, loc=a.loc, nch=a.nch, dropout_cuoi=a.dropout_cuoi, epochs=a.epochs, lo=a.lo,
              ns_list=ns_list, P_hv=P_HV, vong=VONG, buoc=BUOC, giay_huan_luyen=t_hl, giay=time.time() - t0, **res)
    json.dump(js, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print(f"fold {i+1}: tinh {res['tinh_giu_lai']:.4f} " + " ".join(f"{m} {np.round(res[m], 3).tolist()}" for m in MODES) + f" {time.time()-t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--nch", type=int, default=22)
    ap.add_argument("--dropout-cuoi", type=float, default=0.95)
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--cho-phep-cpu", action="store_true")
    a = ap.parse_args()
    a.loc = "causal"; ds = "benchmark"
    if a.epochs < 1 or a.lo < 1 or a.nch < 1 or not (0 < a.T <= 1.0) or a.subjects < 0:
        sys.exit("tham so khong hop le")
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = a.subjects or c["n"]
    try:
        if a.folds == "all":
            folds = list(range(1, nsub + 1))
        elif "::" in a.folds:
            i0, k = (int(v) for v in a.folds.split("::")); assert k >= 1; folds = list(range(i0, nsub + 1, k))
        else:
            i0, i1 = (int(v) for v in a.folds.split(":")); folds = list(range(i0, i1 + 1))
    except (ValueError, AssertionError):
        sys.exit("--folds khong hop le")
    if not folds or any(not (1 <= f <= nsub) for f in folds):
        sys.exit("--folds ngoai mien")
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(seed=SEED, ds=ds, loc=a.loc, T=a.T, epochs=a.epochs, lo=a.lo, nch=a.nch, dropout_cuoi=a.dropout_cuoi, subjects=a.subjects,
               ns_list=NS_LIST, P_hv=P_HV, vong=VONG, buoc=BUOC, modes=MODES,
               script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
               global_dnn_sha256=hashlib.sha256(open(GD.__file__, "rb").read()).hexdigest(),
               b1_diem_sha256=hashlib.sha256(open(B.__file__, "rb").read()).hexdigest())
    cfgf = os.path.join(a.out, "config.json")
    if os.path.exists(cfgf):
        if json.load(open(cfgf)) != cfg:
            sys.exit("config khac lan chay truoc: khong tron ket qua")
    else:
        tmp = f"{cfgf}.{os.getpid()}.tmp"; json.dump(cfg, open(tmp, "w"), indent=1); os.replace(tmp, cfgf)
    D = {"X": [], "y": []}
    for s in range(1, nsub + 1):
        X, y, _ = B.load_subject(ds, s, B.CH9)
        if X.shape[0] != 240 or X.shape[1] != 9 or not np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, 6)):
            sys.exit(f"du lieu nguoi {s} sai cau truc")
        D["X"].append(GD.tien_xu_ly(X, ds, a.loc, ns, fs)); D["y"].append(y)
    print(f"nap {nsub} nguoi, T = {a.T}, thiet bi {dev}", flush=True)
    for f in folds:
        chay_fold(f - 1, D, a, dev)


if __name__ == "__main__":
    main()
