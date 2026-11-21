#!/usr/bin/env python
"""TAI HIEN Ensemble-DNN (guney2023transfer, J. Neural Eng. 2023, DOI 10.1088/1741-2552/acacca) theo MA GOC osmanberke/Ensemble-of-DNNs
(commit aabd8e4: training.m, testing.m), tren giao thuc cua du an (loc NHAN QUA, LOSO, thu tu tep).
Bien the:
  du : giai doan 1 = Global DNN 120 kenh dac trung hoc tren moi nguoi nguon (Adam 1e-4, 1000 vong Benchmark / 800 BETA, lo 100, L2 1e-3 tren trong so).
  22 : giai doan 1 = DUNG LAI mang nen 22 kenh cua KTN (--trongso), khong hoc lai.
Giai doan 2 (ca hai): moi nguoi nguon, tinh chinh TOAN MANG tu mang giai doan 1 tren du lieu co nhan cua nguoi do: Adam 1e-4, 1000 vong,
  lo 40 x (so khoi - 1), bo lo cuoi le (nhu MATLAB), L2 1e-3 tren trong so, dropout hai lop dau 0,6 (Benchmark) / 0,7 (BETA), dropout cuoi 0,95.
Moi luot cua nguoi dung (khong nhan, khong tich luy): moi mang nguon du doan; do giong cua nguon n = max tren cac bo loc khong gian (cot cua l2)
  cua [corr^2(w^T mau_lop_du_doan, w^T x) + CCA^2(w^T x, tham chieu 5 hai cua lop du doan)]; mau = TB qua khoi du lieu nguon, gop dai con bang
  trong so l1 cua mang nguon n; x cung gop dai con bang trong so do. Xep nguon giam dan; voi moi k bo phieu co trong so do giong; chon k co le
  (phieu lon nhat - phieu thu hai) lon nhat (k nho nhat neu bang); quyet dinh = lop nhieu phieu nhat o k do.
Cham: luot 41..het ("sau", cung tap voi KTN) va moi luot ("phien"). Moi fold mot JSON (chay song song --folds k::4).
"""
import argparse, copy, json, os, sys, time, hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B                  # noqa: E402
import global_dnn as GD              # noqa: E402
import global_dnn_thichnghi as GT    # noqa: E402

SEED = 20261107
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
N0 = 40
CFG = {"benchmark": dict(khoi=6, vong1=1000, do2=0.6), "beta": dict(khoi=4, vong1=800, do2=0.7)}


def tinh_chinh(m0, X, y, khoi, do2, dev, seed, vong=1000):
    m = copy.deepcopy(m0)
    m.d1.p = do2; m.d2.p = do2; m.d3.p = 0.95
    for p in m.parameters():
        p.requires_grad_(True)
    ps = list(m.named_parameters())
    opt = torch.optim.Adam([{"params": [p for k, p in ps if k.endswith("weight")], "weight_decay": 1e-3},
                            {"params": [p for k, p in ps if k.endswith("bias")], "weight_decay": 0.0}], lr=1e-4)
    lo = GD.NT * (khoi - 1); n = len(y); nb = n // lo
    g = torch.Generator(device=dev); g.manual_seed(seed)
    lossf = nn.CrossEntropyLoss()
    for _ in range(vong):
        m.train()
        perm = torch.randperm(n, device=dev, generator=g)
        for k in range(nb):
            idx = perm[k * lo:(k + 1) * lo]
            opt.zero_grad(set_to_none=True)
            lossf(m(X[idx]), y[idx]).backward(); opt.step()
    m.eval()
    return m


def tham_chieu(freqs, fs, ns, dev):
    t = np.arange(ns) / fs
    Q = []
    for f in freqs:
        Y = np.stack([fn(2 * np.pi * h * f * t) for h in range(1, 6) for fn in (np.sin, np.cos)], 1)
        Q.append(np.linalg.qr(Y - Y.mean(0))[0])
    return torch.tensor(np.stack(Q), dtype=torch.float32, device=dev)          # (N, ns, 10), da tru TB (CCA co tru TB)


def do_giong(m, Xs, ys, Xu, pred, Qr):
    """m: mang nguon; Xs (n_s, NSB, C, ns) du lieu nguon; Xu (n, NSB, C, ns); pred (n,) du doan cua m. Tra (n,) do giong."""
    wb = m.l1.weight.reshape(-1)                                               # (NSB,)
    W = m.l2.weight[:, 0, :, 0]                                                # (K, C)
    mau = torch.stack([Xs[ys == k].mean(0) for k in range(GD.NT)])             # (N, NSB, C, ns)
    mau = torch.einsum("b,nbct->nct", wb, mau)                                 # (N, C, ns)
    xu = torch.einsum("b,ibct->ict", wb, Xu)                                   # (n, C, ns)
    zt = torch.einsum("kc,ict->ikt", W, mau[pred])                             # (n, K, ns) mau lop du doan chieu qua tung bo loc
    zx = torch.einsum("kc,ict->ikt", W, xu)                                    # (n, K, ns)
    zt = zt - zt.mean(-1, keepdim=True); zx = zx - zx.mean(-1, keepdim=True)
    r = (zt * zx).sum(-1) / torch.sqrt((zt * zt).sum(-1) * (zx * zx).sum(-1) + 1e-30)
    pr = torch.einsum("ikt,itq->ikq", zx, Qr[pred])                            # CCA 1 chieu - 10 chieu = R^2 hoi quy
    cca2 = (pr ** 2).sum(-1) / ((zx ** 2).sum(-1) + 1e-30)
    return (r ** 2 + cca2).max(1).values


def chon_dong(sim, pred):
    """sim, pred: (S, n). Tra (n,) quyet dinh theo bo phieu k dong nhu testing.m."""
    S, n = sim.shape
    od = torch.argsort(sim, dim=0, descending=True, stable=True)
    ss = torch.gather(sim, 0, od); pp = torch.gather(pred, 0, od)              # (S, n)
    phieu = torch.zeros(n, GD.NT, device=sim.device)
    tot_le = torch.full((n,), -1e30, device=sim.device); kq = torch.zeros(n, dtype=torch.long, device=sim.device); k_chon = torch.zeros(n, dtype=torch.long, device=sim.device)
    for k in range(S):
        phieu.scatter_add_(1, pp[k][:, None], ss[k][:, None])
        top2 = torch.topk(phieu, 2, dim=1).values
        le = top2[:, 0] - top2[:, 1]
        moi = le > tot_le                                                      # MATLAB max: lay k nho nhat neu bang
        tot_le = torch.where(moi, le, tot_le)
        kq = torch.where(moi, phieu.argmax(1), kq); k_chon = torch.where(moi, torch.full_like(k_chon, k + 1), k_chon)
    return kq, k_chon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta"])
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--bien-the", required=True, choices=["du", "22"])
    ap.add_argument("--trongso", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--vong1", type=int, default=0, help="chi de chay thu; 0 = theo ma goc")
    ap.add_argument("--vong2", type=int, default=0, help="chi de chay thu; 0 = 1000")
    ap.add_argument("--cho-phep-cpu", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    cf = CFG[a.ds]; c = B.DS[a.ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    if a.bien_the == "22":
        cfgw = json.load(open(os.path.join(a.trongso, "config.json")))
        assert cfgw["nch"] == 22 and cfgw["loc"] == "causal" and abs(cfgw["T"] - a.T) < 1e-9 and cfgw.get("ds", "benchmark") == a.ds
        assert cfgw["subjects"] == 0 and cfgw["epochs"] == 1000 and cfgw["lo"] == 100 and abs(cfgw["dropout_cuoi"] - 0.95) < 1e-9 and cfgw["seed"] == SEED  # dung mang nen cua KTN
    if a.folds == "all":
        folds = list(range(1, nsub + 1))
    else:
        i0, k = (int(v) for v in a.folds.split("::")); folds = list(range(i0, nsub + 1, k))
    os.makedirs(a.out, exist_ok=True)
    D = {"X": [], "y": []}; freqs = None
    nl = cf["khoi"] * GD.NT
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(a.ds, s, B.CH9)
        assert X.shape[0] == nl and np.isfinite(X).all() and np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, cf["khoi"]))
        freqs = np.asarray(fr, float) if freqs is None else freqs
        D["X"].append(torch.tensor(GD.tien_xu_ly(X, a.ds, "causal", ns, fs), device=dev)); D["y"].append(torch.tensor(y, device=dev, dtype=torch.long))
    Qr = tham_chieu(freqs, fs, ns, dev)
    print(f"nap {nsub} nguoi ({a.ds}), T = {a.T}, bien the {a.bien_the}, {dev}", flush=True)
    me = sha(os.path.abspath(__file__))
    phu = {os.path.basename(m_.__file__): sha(m_.__file__) for m_ in (GD, GT, B)}                  # hash ma phu
    cfg = dict(ds=a.ds, T=a.T, bien_the=a.bien_the, trongso=os.path.abspath(a.trongso) if a.trongso else "", vong1=a.vong1, vong2=a.vong2, seed=SEED,
               script_sha256=me, ma_phu_sha256=phu)
    cfgf = os.path.join(a.out, "config.json")
    if os.path.exists(cfgf):
        if json.load(open(cfgf)) != cfg:
            sys.exit("config khac lan chay truoc: khong tron ket qua")
    else:
        tmp = f"{cfgf}.{os.getpid()}.tmp"; json.dump(cfg, open(tmp, "w"), indent=1); os.replace(tmp, cfgf)
    for f in folds:
        i = f - 1; fn = os.path.join(a.out, f"fold{f:03d}.json")
        if os.path.exists(fn):
            continue
        seed = SEED + 1000 * f + int(round(a.T * 10))
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        ck_sha = None
        src = [s for s in range(nsub) if s != i]
        t0 = time.time()
        if a.bien_the == "du":
            Xtr = torch.cat([D["X"][s] for s in src]); ytr = torch.cat([D["y"][s] for s in src])
            ga = argparse.Namespace(nch=120, dropout_cuoi=0.95, epochs=a.vong1 or cf["vong1"], lo=100)
            m1 = GT.huan_luyen(Xtr, ytr, Xtr.shape[2], Xtr.shape[3], ga, dev, seed); del Xtr, ytr
        else:
            m1 = GD.GlobalDNN(9, ns, nch=22, p_cuoi=0.95).to(dev)
            ck = os.path.join(a.trongso, f"fold{f:03d}_trongso.pt")
            m1.load_state_dict(torch.load(ck, map_location=dev)); ck_sha = sha(ck)
        m1.eval(); t1 = time.time() - t0
        preds, sims = [], []
        Xu, yu = D["X"][i], D["y"][i]
        for j, s in enumerate(src):
            ms = tinh_chinh(m1, D["X"][s], D["y"][s], cf["khoi"], cf["do2"], dev, seed + 17 * (j + 1), a.vong2 or 1000)
            with torch.no_grad():
                pr = ms(Xu).argmax(1)
                preds.append(pr); sims.append(do_giong(ms, D["X"][s], D["y"][s], Xu, pr, Qr))
            del ms
        P = torch.stack(preds); Sm = torch.stack(sims)                          # (S, n)
        dec, kc = chon_dong(Sm, P)
        ok = (dec == yu).float().cpu().numpy()
        with torch.no_grad():
            tinh = (m1(Xu).argmax(1) == yu).float().cpu().numpy()
        js = dict(fold=f, seed=seed, ds=a.ds, T=a.T, bien_the=a.bien_the, nch=120 if a.bien_the == "du" else 22, n0=N0,
                  sau=float(ok[N0:].mean()), phien=float(ok.mean()), k_tb=float(kc.float().mean()), k_tb_sau=float(kc[N0:].float().mean()),
                  mang_giai_doan1_sau=float(tinh[N0:].mean()), mang_giai_doan1_phien=float(tinh.mean()),
                  nguon_dung_TB=float((P[:, N0:] == yu[None, N0:]).float().mean()), giay_giai_doan1=t1, giay=time.time() - t0,
                  vong1=(a.vong1 or cf["vong1"]) if a.bien_the == "du" else None, vong2=a.vong2 or 1000, script_sha256=me, ma_phu_sha256=phu, trongso_sha256=ck_sha)
        json.dump(js, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
        print(f"fold {f}: xong {time.time()-t0:.0f} s", flush=True)                # khong in do chinh xac


if __name__ == "__main__":
    main()
