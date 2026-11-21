#!/usr/bin/env python
"""B1 buoc 1: tinh diem tung nhanh (atom = phuong phap x dai loc) cho moi luot thu, LOSO khong hieu chuan.

Ma viet moi cho du an nay (NCS Ton B1); khong dung ma cua ssvep_lite.

Nhanh (moi dai loc m = 1..5 la mot atom rieng):
  fbcca : tuong quan chinh tac lon nhat giua X va tham chieu sin-cos (H hai)           [Chen 2015]
  ttcca : CCA mau chuyen giao (template = trung binh luot cua nguoi nguon), diem
          sum sign(r) r^2 voi r1 = CCA(X,Y), r2 = corr qua w tu CCA(X, Xbar),
          r3 = corr qua w tu CCA(X, Y)                                              [Yuan 2015]
  fixf  : bo loc khong gian co dinh hoc giua nguoi (TRCA tren trung binh tung nguoi nguon,
          ensemble 40 bo loc), diem = tuong quan W^T X voi W^T Xbar_k
Dau ra: results/<out>/<ds>_scores_w<T>.npz (z-score 40 muc tieu moi luot, nhan, nguoi) va
        <ds>_chan_acc.csv (do chinh xac FB theo tap kenh, cho B1-H1).
"""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.io import loadmat
from scipy.signal import cheb1ord, cheby1, sosfilt

SEED = 20261105
np.random.seed(SEED)

DS = {
    "benchmark": dict(root="/gpfs0/users/gv_dnton/datasets/ssvep/benchmark", n=35, fs=250, onset=125, lat=0.14),
    "beta": dict(root="/gpfs0/users/gv_nphuy/datasets/BETA/raw", n=70, fs=250, onset=125, lat=0.13),
}
CH9 = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
CHSETS = {"c9": CH9, "c6": ["PO3", "POZ", "PO4", "O1", "OZ", "O2"], "c3": ["O1", "OZ", "O2"]}
NB, H, NTGT = 5, 5, 40
FBW = np.array([(m + 1) ** -1.25 + 0.25 for m in range(NB)])
WINS = [0.2, 0.4, 0.6, 0.8, 1.0]
WINS_CHAN = [0.4, 1.0]
METHODS = ["fbcca", "ttcca", "fixf"]


def filterbank(fs):
    sos = []
    for m in range(1, NB + 1):
        wp = [8 * m / (fs / 2), 88 / (fs / 2)]
        ws = [(8 * m - 2) / (fs / 2), 100 / (fs / 2)]
        N, Wn = cheb1ord(wp, ws, 3, 40)
        sos.append(cheby1(N, 0.5, Wn, btype="bandpass", output="sos"))
    return sos


def load_subject(ds, s, chans):
    c = DS[ds]
    if ds == "benchmark":
        d = loadmat(os.path.join(c["root"], f"S{s}.mat"))["data"]          # ch,t,target,block
        names = [l.split()[3].upper() for l in open(os.path.join(c["root"], "64-channels.loc")) if l.strip()]
        fp = loadmat(os.path.join(c["root"], "Freq_Phase.mat"))
        freqs = np.asarray(fp["freqs"]).ravel()
        X = d.transpose(3, 2, 0, 1)                                          # block,target,ch,t
    else:
        dd = loadmat(os.path.join(c["root"], f"S{s}.mat"), squeeze_me=True, struct_as_record=False)["data"]
        names = [str(r[3]).strip().upper() for r in dd.suppl_info.chan]
        freqs = np.asarray(dd.suppl_info.freqs).ravel()
        X = dd.EEG.transpose(2, 3, 0, 1)                                     # block,target,ch,t
    idx = [names.index(ch) for ch in chans]
    X = X[:, :, idx, :].astype(np.float64)
    nb, nt = X.shape[:2]
    y = np.tile(np.arange(nt), nb)
    return X.reshape(nb * nt, len(idx), -1), y, freqs


def preprocess(X, ds, sos, ns_max):
    """Tru TB doan truoc kich thich, loc nhan qua tu mau 0, cat [onset+lat, +ns_max). Ra (trial, band, C, ns_max)."""
    c = DS[ds]
    on = c["onset"]; st = on + int(round(c["lat"] * c["fs"]))
    X = X[:, :, : st + ns_max] - X[:, :, :on].mean(axis=2, keepdims=True)
    out = np.empty((X.shape[0], NB, X.shape[1], ns_max))
    for m, so in enumerate(sos):
        out[:, m] = sosfilt(so, X, axis=2)[:, :, st: st + ns_max]
    return out


def refs(freqs, fs, ns):
    t = np.arange(ns) / fs
    Y = np.stack([np.concatenate([[np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)] for h in range(1, H + 1)])
                  for f in freqs])                                            # (N, 2H, ns)
    Y = Y - Y.mean(-1, keepdims=True)
    return np.stack([np.linalg.qr(y.T)[0] for y in Y])                       # (N, ns, 2H)


def center(A):
    return A - A.mean(-1, keepdims=True)


def corr_rows(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-300)


def trca_filters(M):
    """M: (nsub, N, C, ns) trung binh tung nguoi nguon trong mot dai. Tra W (C, N): bo loc TRCA tung muc tieu,
    coi trung binh tung nguoi la mot 'luot' (do lap lai giua nguoi)."""
    nsub, N, C, ns = M.shape
    W = np.empty((C, N))
    for k in range(N):
        A = M[:, k]                                                           # (nsub, C, ns)
        Ssum = A.sum(0)
        S = Ssum @ Ssum.T - np.einsum("ict,idt->cd", A, A)
        Q = np.einsum("ict,idt->cd", A, A)
        ev, evec = np.linalg.eig(np.linalg.solve(Q + 1e-9 * np.trace(Q) / C * np.eye(C), S))
        W[:, k] = np.real(evec[:, np.argmax(np.real(ev))])
    return W


def score_trial(x, Qy, tmpl, Qt, Wf, WT):
    """x: (B, C, ns) da can giua. Tra dict method -> (B, N) diem tung dai."""
    B, C, ns = x.shape
    out = {m: np.empty((B, NTGT)) for m in METHODS}
    for b in range(B):
        Qx, Rx = np.linalg.qr(x[b].T)                                         # (ns,C),(C,C)
        # fbcca + r1, w_xy
        Mxy = np.einsum("tc,ktj->kcj", Qx, Qy)                                # (N,C,2H)
        U, Sv, _ = np.linalg.svd(Mxy)
        r1 = Sv[:, 0]
        out["fbcca"][b] = r1
        # ttcca
        Mxt = np.einsum("tc,ktd->kcd", Qx, Qt[b])                             # (N,C,C)
        U2, S2, _ = np.linalg.svd(Mxt)
        wx = np.linalg.solve(Rx, U2[:, :, 0].T).T                             # (N,C)
        wxy = np.linalg.solve(Rx, U[:, :, 0].T).T
        T = tmpl[b]                                                           # (N,C,ns)
        r2 = corr_rows(np.einsum("kc,ct->kt", wx, x[b]), np.einsum("kc,kct->kt", wx, T))
        r3 = corr_rows(np.einsum("kc,ct->kt", wxy, x[b]), np.einsum("kc,kct->kt", wxy, T))
        out["ttcca"][b] = sum(np.sign(r) * r ** 2 for r in (r1, r2, r3))
        # fixf
        p = (Wf[b].T @ x[b]).ravel()                                          # (N*ns,)
        out["fixf"][b] = corr_rows(p[None, :], WT[b].reshape(NTGT, -1))
    return out


def macs(method, C, ns, n=NTGT, h2=2 * H):
    """Phep nhan chinh moi quyet dinh, MOT dai (khong tinh chuan hoa, can, bank loc). Theo cach hien thuc trong tep nay."""
    qr = 2 * ns * C * C
    if method == "fbcca":
        return qr + n * (ns * C * h2 + C * h2 * h2)
    if method == "ttcca":
        return qr + n * (ns * C * h2 + C * h2 * h2 + ns * C * C + C ** 3 + 2 * C * C + 4 * ns * C + 4 * ns)
    if method == "fixf":
        return n * C * ns + n * (n * ns) + n * ns
    raise ValueError(method)


def bank_macs(sos, C, n_samples):
    return sum(5 * so.shape[0] for so in sos) * C * n_samples


G = {}


def work(i):
    """Diem cua nguoi i (LOSO: mau tu moi nguoi khac) o moi cua so."""
    P, ys, freqs, fs, wins, chidx = G["P"], G["y"], G["freqs"], G["fs"], G["wins"], G["chidx"]
    res = {}
    for T in wins:
        ns = int(round(T * fs))
        Qy = refs(freqs, fs, ns)
        x = center(P[i][:, :, chidx, :ns])                                     # (trial,B,C,ns)
        # mau LOSO: trung binh tung nguoi -> bo nguoi i
        Ms = G["means"][T]                                                     # (nsub,B,N,C,ns) da can giua
        others = [j for j in range(len(P)) if j != i]
        Mo = Ms[others][:, :, :, chidx, :]
        tmpl = Mo.mean(0)                                                      # (B,N,C,ns)
        Qt = np.stack([[np.linalg.qr(tmpl[b, k].T)[0] for k in range(NTGT)] for b in range(NB)])  # (B,N,ns,C)
        Wf = np.stack([trca_filters(Mo[:, b]) for b in range(NB)])           # (B,C,N)
        WT = np.einsum("bck,bnct->bnkt", Wf, tmpl)                            # (B,N,N,ns)
        sc = np.empty((len(METHODS), NB, x.shape[0], NTGT))
        for t in range(x.shape[0]):
            o = score_trial(x[t], Qy, tmpl, Qt, Wf, WT)
            for mi, m in enumerate(METHODS):
                sc[mi, :, t] = o[m]
        res[T] = sc
    return i, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(DS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=0, help="0 = tat ca; >0 = chi n nguoi dau (smoke)")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = DS[a.ds]; fs = c["fs"]; sos = filterbank(fs)
    nsub = a.subjects or c["n"]
    ns_max = int(round(max(WINS) * fs))
    t0 = time.time()
    P, Y = [], []
    freqs = None
    for s in range(1, nsub + 1):
        X, y, freqs = load_subject(a.ds, s, CH9)
        P.append(preprocess(X, a.ds, sos, ns_max)); Y.append(y)
    print(f"[{a.ds}] nap + tien xu ly {nsub} nguoi: {time.time()-t0:.1f} s", flush=True)
    cfg = dict(seed=SEED, ds=a.ds, n_subjects=nsub, fs=fs, onset=c["onset"], latency_s=c["lat"], channels=CH9,
               chansets=CHSETS, bands=[[8 * m, 88] for m in range(1, NB + 1)], cheby1_rp=0.5, H=H, fb_weights=FBW.tolist(),
               windows=WINS, methods=METHODS, preprocessing="tru TB doan truoc kich thich; loc nhan qua sosfilt tu mau 0; cat [onset+lat, +T)",
               protocol="LOSO, khong hieu chuan; mau va bo loc fixf tu moi nguoi khac",
               macs_per_band={m: {str(T): {cs: macs(m, len(v), int(round(T * fs))) for cs, v in CHSETS.items()} for T in WINS} for m in METHODS},
               bank_macs={str(T): bank_macs(sos, 9, c["onset"] + int(round(c["lat"] * fs)) + int(round(T * fs))) for T in WINS},
               sos_sections=[int(so.shape[0]) for so in sos])
    json.dump(cfg, open(os.path.join(a.out, f"{a.ds}_config.json"), "w"), indent=1, ensure_ascii=False)

    def run(chname, wins):
        chidx = [CH9.index(ch) for ch in CHSETS[chname]]
        means = {}
        for T in wins:
            ns = int(round(T * fs))
            means[T] = np.stack([np.stack([[center(P[s][Y[s] == k][:, b, :, :ns]).mean(0) for k in range(NTGT)]
                                           for b in range(NB)]) for s in range(nsub)])   # (nsub,B,N,C9,ns)
        G.update(P=P, y=Y, freqs=freqs, fs=fs, wins=wins, chidx=chidx, means=means)
        with Pool(a.procs) as pool:
            out = dict(pool.map(work, range(nsub)))
        return out

    # 1) atom diem, 9 kenh, moi cua so
    out = run("c9", WINS)
    for T in WINS:
        sc = np.stack([out[i][T] for i in range(nsub)], axis=0)             # (nsub, M, B, trial, N)
        z = (sc - sc.mean(-1, keepdims=True)) / (sc.std(-1, keepdims=True) + 1e-12)
        np.savez_compressed(os.path.join(a.out, f"{a.ds}_scores_w{T:.1f}.npz"), z=z.astype(np.float32),
                            raw=sc.astype(np.float32), y=np.stack(Y), methods=np.array(METHODS))
    print(f"[{a.ds}] diem 9 kenh xong: {time.time()-t0:.1f} s", flush=True)
    # 2) do chinh xac FB theo tap kenh (B1-H1)
    rows = ["ds,chanset,window,method,subject,acc"]
    for chname in CHSETS:
        o = out if chname == "c9" else run(chname, WINS_CHAN)
        for T in WINS_CHAN:
            for i in range(nsub):
                sc = o[i][T]
                for mi, m in enumerate(METHODS):
                    s_b = sc[mi]
                    fb = (FBW[:, None, None] * (s_b ** 2 if m == "fbcca" else s_b)).sum(0)
                    acc = (fb.argmax(-1) == Y[i]).mean() * 100
                    rows.append(f"{a.ds},{chname},{T},{m},{i+1},{acc:.4f}")
        print(f"[{a.ds}] tap kenh {chname} xong: {time.time()-t0:.1f} s", flush=True)
    open(os.path.join(a.out, f"{a.ds}_chan_acc.csv"), "w").write("\n".join(rows) + "\n")
    print(f"[{a.ds}] DONE {time.time()-t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
