#!/usr/bin/env python
"""Tai hien Global DNN (guney2022deep; giai doan 1 cua guney2023transfer) cho B1.

Chuyen tu ma MATLAB cong khai osmanberke/Ensemble-of-DNNs (commit aabd8e4b62dccb0e57d8d85c3c2888e80050682b): PreProcess.m, training.m.

Tien xu ly (--loc):
  filtfilt : nhu PreProcess.m: cat doan [onset + do tre, + N) (onset 0,5 s, do tre 0,14 s, Benchmark), KHONG tru nen, loc hai chieu
             tren CHINH doan da cat (scipy filtfilt, padtype mac dinh; MATLAB filtfilt khac chut o xu ly bien).
  causal   : cung giao thuc B1 (b1_diem.preprocess): tru TB doan truoc kich thich, loc nhan qua (sosfilt) tu mau 0, cat cung doan.
  Ca hai: 3 dai con, Chebyshev I, bac bang thong 2 (designfilt 'FilterOrder' 2 = cheby1 N = 1), gon 1 dB, dai [8 r, 90] Hz, fs 250.
Mang (training.m): conv 1x1 gop dai (trong so khoi tao 1, khong bias), conv Cx1 120 bo, dropout 0,1, conv 1x2 buoc 2 120->120,
  dropout 0,1, ReLU, conv 1x10 'same' 120->120, dropout 0,95, FC 40. Khoi tao 'narrow-normal' N(0, 0,01), bias 0.
Huan luyen: Adam lr 1e-4 khong giam, L2 1e-3 (chi trong so, nhu MATLAB BiasL2Factor = 0), lo 100, xao moi epoch, bo lo le cuoi
  (nhu trainNetwork), 1000 epoch (Benchmark), khong dung som. Chi giai doan 1 (Global DNN), khong tinh chinh.
LOSO: hoc tren 34 nguoi, thu tren nguoi con lai (moi luot cua nguoi do). Moi fold ghi mot JSON; khong ghi de.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import cheby1, filtfilt, sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B  # noqa: E402

SEED = 20261107
NSB, NCH, NT = 3, 120, 40


def bo_loc(fs):
    ba, sos = [], []
    for r in range(1, NSB + 1):
        Wn = [8 * r / (fs / 2), 90 / (fs / 2)]
        ba.append(cheby1(1, 1, Wn, btype="bandpass", output="ba"))
        sos.append(cheby1(1, 1, Wn, btype="bandpass", output="sos"))
    return ba, sos


def tien_xu_ly(X, ds, loc, ns, fs):
    """X (luot, C, t) tho. Tra (luot, NSB, C, ns) float32."""
    c = B.DS[ds]; on = c["onset"]; st = on + int(round(c["lat"] * fs))
    ba, sos = bo_loc(fs)
    out = np.empty((X.shape[0], NSB, X.shape[1], ns), np.float32)
    if loc == "filtfilt":
        seg = X[:, :, st: st + ns]
        for r, (b, a) in enumerate(ba):
            out[:, r] = filtfilt(b, a, seg, axis=2, padtype="odd", padlen=3 * (max(len(a), len(b)) - 1))   # nhu MATLAB: nfact = 3*(nfilt-1)
    else:
        Xb = X[:, :, : st + ns] - X[:, :, :on].mean(axis=2, keepdims=True)
        for r, so in enumerate(sos):
            out[:, r] = sosfilt(so, Xb, axis=2)[:, :, st: st + ns]
    return out


class GlobalDNN(nn.Module):
    def __init__(self, C, ns, nclass=NT, nch=NCH, p_cuoi=0.95):
        super().__init__()
        self.l1 = nn.Conv2d(NSB, 1, (1, 1), bias=False)
        self.l2 = nn.Conv2d(1, nch, (C, 1))
        self.d1 = nn.Dropout(0.1)
        self.l3 = nn.Conv2d(nch, nch, (1, 2), stride=(1, 2))
        self.d2 = nn.Dropout(0.1)
        self.l4 = nn.Conv2d(nch, nch, (1, 10), padding="same")
        self.d3 = nn.Dropout(p_cuoi)
        self.fc = nn.Linear(nch * (ns // 2), nclass)
        nn.init.ones_(self.l1.weight)
        for m in (self.l2, self.l3, self.l4, self.fc):
            nn.init.normal_(m.weight, 0.0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):                      # x (B, NSB, C, ns)
        h = self.l2(self.l1(x))
        h = torch.relu(self.d2(self.l3(self.d1(h))))
        h = self.d3(self.l4(h))
        return self.fc(h.flatten(1))


def dem_mac(model, C, ns, nch=NCH):
    """MAC giai tich theo lop (khong gom bank loc)."""
    n2 = ns // 2
    m = {"l1": NSB * C * ns, "l2": nch * C * ns, "l3": nch * n2 * nch * 2, "l4": nch * n2 * nch * 10, "fc": nch * n2 * NT}
    m["tong"] = sum(m.values())
    return m


def chay_fold(i, D, a, dev):
    fn = os.path.join(a.out, f"fold{i+1:03d}.json")
    if os.path.exists(fn):
        return
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tr = [s for s in range(len(D["X"])) if s != i]
    Xtr = torch.tensor(np.concatenate([D["X"][s] for s in tr]), device=dev)
    ytr = torch.tensor(np.concatenate([D["y"][s] for s in tr]), device=dev, dtype=torch.long)
    C, ns = Xtr.shape[2], Xtr.shape[3]
    model = GlobalDNN(C, ns, nch=a.nch, p_cuoi=a.dropout_cuoi).to(dev)
    w = [p for n, p in model.named_parameters() if n.endswith("weight")]
    b = [p for n, p in model.named_parameters() if n.endswith("bias")]
    opt = torch.optim.Adam([{"params": w, "weight_decay": 1e-3}, {"params": b, "weight_decay": 0.0}], lr=1e-4)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator(device=dev); g.manual_seed(seed)
    n = len(ytr); nb = n // a.lo
    t0 = time.time(); hist = []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=g)
        tot = 0.0
        for k in range(nb):
            idx = perm[k * a.lo:(k + 1) * a.lo]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step(); tot += float(loss)
        if (ep + 1) % 100 == 0 or ep == 0:
            hist.append((ep + 1, tot / nb))
    model.eval()
    with torch.no_grad():
        Xte = torch.tensor(D["X"][i], device=dev)
        pr = model(Xte).argmax(1).cpu().numpy()
    acc = float((pr == D["y"][i]).mean())
    js = dict(fold=i + 1, seed=seed, T=a.T, loc=a.loc, epochs=a.epochs, lo=a.lo, acc=acc, n_test=int(len(pr)),
              giay=time.time() - t0, loss_hist=hist, mac=dem_mac(model, C, ns, a.nch), nch=a.nch, dropout_cuoi=a.dropout_cuoi, du_doan=pr.tolist())
    json.dump(js, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print(f"fold {i+1}: acc {acc:.4f}, {time.time()-t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="benchmark", choices=["benchmark"])
    ap.add_argument("--loc", required=True, choices=["filtfilt", "causal"])
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="all", help="'all' hoac 'a:b' (dem tu 1, gom b) hoac 'i::k' (bat dau i, buoc k)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--nch", type=int, default=NCH, help="so kenh dac trung (goc 120); ban thu nho de dat ngan sach MAC")
    ap.add_argument("--dropout-cuoi", type=float, default=0.95, help="dropout truoc FC (goc 0,95)")
    ap.add_argument("--cho-phep-cpu", action="store_true", help="chi de thu tren may khong GPU; job chinh thuc bat buoc CUDA")
    a = ap.parse_args()
    if a.epochs < 1 or a.lo < 1 or a.nch < 1 or not (0 <= a.dropout_cuoi < 1) or not (0 < a.T <= 1.0) or a.subjects < 0:
        sys.exit("tham so khong hop le (epochs, lo, T, subjects)")
    os.makedirs(a.out, exist_ok=True)
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    cfgf = os.path.join(a.out, "config.json")
    cfg = dict(seed=SEED, ds=a.ds, loc=a.loc, T=a.T, epochs=a.epochs, lo=a.lo, subjects=a.subjects, nch=a.nch, dropout_cuoi=a.dropout_cuoi, script_sha256=me,
               b1_diem_sha256=hashlib.sha256(open(B.__file__, "rb").read()).hexdigest(), torch=torch.__version__,
               nguon_ma="osmanberke/Ensemble-of-DNNs@aabd8e4b62dccb0e57d8d85c3c2888e80050682b training.m, PreProcess.m")
    el = os.path.join(a.out, "env.lock"); os.makedirs(el, exist_ok=True)
    if not os.path.exists(os.path.join(el, "pip-freeze.txt")):
        import subprocess, scipy
        open(os.path.join(el, f"pip-freeze.{os.getpid()}.tmp"), "w").write(
            subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
        os.replace(os.path.join(el, f"pip-freeze.{os.getpid()}.tmp"), os.path.join(el, "pip-freeze.txt"))
        info = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    cudnn_deterministic=torch.backends.cudnn.deterministic)
        json.dump(info, open(os.path.join(el, "moi-truong.json"), "w"), indent=1)
    if os.path.exists(cfgf):
        old = json.load(open(cfgf))
        if {k: old.get(k) for k in cfg if k != "torch"} != {k: cfg[k] for k in cfg if k != "torch"}:
            sys.exit("config khac lan chay truoc trong cung thu muc: khong tron ket qua")
    else:
        tmp = f"{cfgf}.{os.getpid()}.tmp"                      # ghi nguyen tu: tien trinh song song khong doc tep dang ghi do
        json.dump(cfg, open(tmp, "w"), indent=1); os.replace(tmp, cfgf)
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA: job chinh thuc khong duoc chay CPU")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = B.DS[a.ds]; fs = c["fs"]; ns = int(round(a.T * fs))
    nsub = a.subjects or c["n"]
    # chon fold (dem tu 1) va kiem mien TRUOC khi nap du lieu
    try:
        if a.folds == "all":
            folds = list(range(1, nsub + 1))
        elif "::" in a.folds:
            i0, k = (int(v) for v in a.folds.split("::")); assert k >= 1; folds = list(range(i0, nsub + 1, k))
        else:
            i0, i1 = (int(v) for v in a.folds.split(":")); folds = list(range(i0, i1 + 1))
    except (ValueError, AssertionError):
        sys.exit(f"--folds khong hop le: {a.folds}")
    if not folds or any(not (1 <= f <= nsub) for f in folds):
        sys.exit(f"--folds ngoai mien 1..{nsub} hoac rong: {a.folds}")
    D = {"X": [], "y": []}
    t0 = time.time()
    for s in range(1, nsub + 1):
        X, y, _ = B.load_subject(a.ds, s, B.CH9)
        # kiem du lieu: 240 luot, 9 kenh, moi nhan 0..39 dung 6 lan, gia tri huu han
        if X.shape[0] != 240 or X.shape[1] != 9 or not np.array_equal(np.bincount(y, minlength=NT), np.full(NT, 6)):
            sys.exit(f"du lieu nguoi {s} sai cau truc: {X.shape}")
        xp = tien_xu_ly(X, a.ds, a.loc, ns, fs)
        if not np.isfinite(xp).all():
            sys.exit(f"du lieu nguoi {s} co gia tri khong huu han")
        D["X"].append(xp); D["y"].append(y)
    print(f"nap {nsub} nguoi ({a.loc}, T = {a.T}): {time.time()-t0:.0f} s, thiet bi {dev}", flush=True)
    for f in folds:
        i = f - 1
        assert i not in [s for s in range(nsub) if s != i]
        chay_fold(i, D, a, dev)


if __name__ == "__main__":
    main()
