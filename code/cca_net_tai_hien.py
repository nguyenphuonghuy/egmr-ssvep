#!/usr/bin/env python
"""Tai hien CCA-Net (Deng et al., IEEE TIM 74, 2025, DOI 10.1109/TIM.2025.3571139) tu ma chinh thuc commit 1660d42
(tham-chieu-python/cca-net/goc/, KHONG sua; config thay bang shim/config.py).
Moi fold lap lai DUNG trinh tu main.py dong 156-295 (huan luyen tren nguon, thu tren nguoi thu), chi them:
  --che goc : y nhu ma (loc hai chieu tren cua so da cat, do tre cua ma: nak 0,35 s tu dau epoch, benchmark 0,64 s, beta 0,63 s; seed 41 moi fold).
  --che gt  : giao thuc B1: tru TB doan truoc kich thich, cat dai tu mau 0 den het cua so, loc NHAN QUA (lfilter / sosfilt) cung thiet ke bo loc
              cua ma roi giu T mau cuoi; bat dau cua so = onset + round(lat fs) (benchmark 125 + 35, beta 125 + 32, nak 38 + 35);
              seed 20261107 + 1000 fold + round(10 T); cudnn tat dinh. Cac phep KHONG loc (GLST, LST, Ref) dung cua so da tru nen.
  --du-nhanh: (bao kem) forward tra x + x_cca + x_merge nhu bai mo ta (ma goc chi tra x).
Cham: du doan tung luot, xep lai theo thu tu khoi roi lop (chi so = khoi * NT + lop); goc cham moi luot, gt cham luot 41 den het (va bao ca phien).
Bien the luu (ten theo main.py): final (= "Final accuracy", la CCA-Net), CNN, FBTTCCA_new, CCA_new, FBTTCCA_new+CCA_new,
CNN+FBCCA, CNN+FBTTCCA_new, CNN+FBCCA_new, CNN+FBTTCCA_new+FBCCA_new, CNN+FBTTCCA_new+FBCCA, CNN+FBTTCCA.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True, choices=["nak", "benchmark", "beta", "dong"])
ap.add_argument("--T", type=float, required=True)
ap.add_argument("--che", required=True, choices=["goc", "gt"])
ap.add_argument("--folds", default="1::1", help="k::m: chay fold k, k+m, ... (1-based)")
ap.add_argument("--out", required=True)
ap.add_argument("--du-nhanh", action="store_true")
ap.add_argument("--vong", type=int, default=0, help="chi de chay thu: so vong huan luyen (0 = 100 nhu goc)")
a = ap.parse_args()
os.environ["CN_DS"] = {"nak": "0", "benchmark": "1", "beta": "2", "dong": "3"}[a.ds]
os.environ["CN_T"] = repr(a.T)
os.environ["CN_CHE"] = a.che
HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "tham-chieu-python", "cca-net", "shim"); GOCD = os.path.join(HERE, "tham-chieu-python", "cca-net", "goc")
sys.path[:0] = [SHIM, GOCD]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for l in open(os.path.join(GOCD, "SHA256SUMS.txt")):                       # ma goc khong bi sua
    h, f = l.split(); assert sha(os.path.join(GOCD, f.lstrip("*"))) == h, f

import torch                                     # noqa: E402
from scipy.io import loadmat                     # noqa: E402
from scipy import signal                         # noqa: E402
from config import config                        # noqa: E402  (shim)
import function as FN                            # noqa: E402
import FBCCA as FBm                              # noqa: E402
import CCA as CCm                                # noqa: E402
import model as Mm                               # noqa: E402
assert os.path.dirname(os.path.abspath(FN.__file__)) == GOCD and os.path.dirname(os.path.abspath(sys.modules["config"].__file__)) == SHIM

SEED = 20261107
DS = {"nak": dict(root="/gpfs0/users/gv_dnton/datasets/ssvep/nakanishi2015", n=10, key="eeg", fs=256, ld_goc=int(0.35 * 256), onset=38, lat=0.135,
                  ch=list(range(8)), ten=lambda s: f"s{s}.mat"),
      "benchmark": dict(root="/gpfs0/users/gv_dnton/datasets/ssvep/benchmark", n=35, key="data", fs=250, ld_goc=int((0.14 + 0.5) * 250), onset=125, lat=0.14,
                        ch=[47, 53, 54, 55, 56, 57, 60, 61, 62], ten=lambda s: f"S{s}.mat"),
      "beta": dict(root="/gpfs0/users/gv_nphuy/datasets/BETA/raw", n=70, key="data", fs=250, ld_goc=int((0.13 + 0.5) * 250), onset=125, lat=0.13,
                   ch=[47, 53, 54, 55, 56, 57, 60, 61, 62], ten=lambda s: f"S{s}.mat"),
      "dong": dict(root="/gpfs0/users/gv_dnton/datasets/ssvep/dong2023", n=59, key="eegdata", fs=250, ld_goc=int((0.16 + 0.5) * 250), onset=125, lat=0.16,
                   ch=list(range(8)), ten=lambda s: f"S{s}.mat")}                 # them 2026-11-17 (Dong 2023; bo nay chi chay che do gt)
c = DS[a.ds]; NT = config.num_class; T = config.T; rfs = config.rfs; nBlock = config.nBlock; Nh = config.Nh
assert rfs == c["fs"]
st_gt = c["onset"] + int(round(c["lat"] * rfs))


# ---------- loc nhan qua cho che do gt (cung thiet ke bo loc cua ma goc, giu T mau cuoi) ----------
def fb_nhan_qua(eeg):
    """thay function.filter_bank: eeg label*block*C*Tdai -> label*block*Nm*C*T."""
    nyq = rfs / 2; passband = [6, 14, 22]; stopband = [4, 10, 16]
    out = np.zeros((eeg.shape[0], eeg.shape[1], config.Nm, eeg.shape[2], T))
    for k in range(config.Nm):
        N, Wn = signal.cheb1ord([passband[k] / nyq, 90 / nyq], [stopband[k] / nyq, 100 / nyq], 3, 40)
        sos = signal.cheby1(N, 0.5, Wn, 'bandpass', output='sos')         # cung bo loc, dang SOS cho on dinh so hoc
        out[:, :, k] = signal.sosfilt(sos, eeg, axis=-1)[..., -T:]
    return out


def fbcca_filterbank_nhan_qua(eeg, fs, idx_fb):
    """thay FBCCA.filterbank: (C, Tdai) -> (C, T)."""
    Nq = fs / 2; passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]; stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    N, Wn = signal.cheb1ord([passband[idx_fb] / Nq, 90 / Nq], [stopband[idx_fb] / Nq, 100 / Nq], 3, 40)
    sos = signal.cheby1(N, 0.5, Wn, 'bandpass', output='sos')
    return signal.sosfilt(sos, eeg, axis=-1)[..., -T:]


def fb_stand_f32(Raw):
    """Bang FN.FB_stand(Raw).astype(float32) tung phan tu (cung preprocessing.scale tren tung lat [j, i], tinh float64 roi ep),
    nhung khong cap them mot mang float64 cung co (bo nho)."""
    out = np.empty(Raw.shape, np.float32)
    for i_stand in range(Raw.shape[1]):
        for j_stand in range(Raw.shape[0]):
            out[j_stand, i_stand] = FN.preprocessing.scale(Raw[j_stand, i_stand], axis=1)
    return out


def tfb_nhan_qua(self, X):
    """thay CCA.FilterBank.transform_filterbank (sosfiltfilt) bang sosfilt, giu T mau cuoi."""
    return np.stack([signal.sosfilt(sos, X, axis=-1)[..., -T:] for sos in self.filterbank])


if a.che == "gt":
    filter_bank = fb_nhan_qua
    FBm.filterbank = fbcca_filterbank_nhan_qua
    CCm.FilterBank.transform_filterbank = tfb_nhan_qua
else:
    filter_bank = FN.filter_bank

if a.du_nhanh:
    def fwd_du(self, x, x_cca):                                              # chep Siamese.forward, chi doi dong return (bo chu thich nhu bai)
        x1 = self.conv1(x); x2 = self.conv2(x1); x2 = self.drop1(x2); x3 = self.conv3(x2); x3 = self.drop2(x3); x3_ = x3; x3 = self.relu(x3)
        x4 = self.conv4(x3); x = self.drop3(x4); x = self.flatten(x); x = self.fc(x)
        x_cca = self.conv1_1(x_cca); x_cca = self.conv2_1(x_cca); x_cca = self.drop1(x_cca); x_merge = self.conv3_2(x_cca + x2)
        x_cca = self.conv3_1(x_cca); x_cca = self.drop2(x_cca); x_merge = self.conv4_2(x_cca + x3_ + x_merge); x_merge = self.drop3(x_merge)
        x_merge = self.flatten(x_merge); x_merge = self.fc_2(x_merge)
        x_cca = self.relu(x_cca); x_cca = self.conv4_1(x_cca); x_cca = self.drop3(x_cca); x_cca = self.flatten(x_cca); x_cca = self.fc_1(x_cca)
        return torch.squeeze(x) + torch.squeeze(x_cca) + torch.squeeze(x_merge)
    Mm.Siamese.forward = fwd_du


def nap(s):
    """-> (cua so label*block*C*T dung cho phep KHONG loc, dau vao cho phep CO loc)."""
    m = loadmat(os.path.join(c["root"], c["ten"](s)))
    d = m[c["key"]]
    if a.ds == "nak":
        d = d.transpose([1, 2, 0, 3])
    elif a.ds == "beta":
        d = d[0][0][0].transpose([0, 1, 3, 2])
    d = d[c["ch"]].astype(np.float64)                                         # [C, Tall, class, block]
    assert d.shape[2] == NT and d.shape[3] == nBlock and np.isfinite(d).all(), (s, d.shape)
    if a.che == "goc":
        w = d[:, c["ld_goc"]:c["ld_goc"] + T].transpose(2, 3, 0, 1)
        assert w.shape[-1] == T
        return w, w
    dl = d[:, :st_gt + T] - d[:, :c["onset"]].mean(axis=1, keepdims=True)
    dl = dl.transpose(2, 3, 0, 1)
    assert dl.shape[-1] == st_gt + T
    return dl[..., st_gt:], dl


def dat_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)


def chay_fold(f, D):
    t0 = time.time()
    seed = 41 if a.che == "goc" else SEED + 1000 * f + int(round(a.T * 10))
    if a.che == "gt":
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    src = [s for s in range(1, c["n"] + 1) if s != f]
    nsrc = len(src)
    trd, trw, trl, trlong = [], [], [], []
    for s in src:                                                             # main.py 156-187
        w, L = D[s]
        fbd = filter_bank(L)
        lst = np.zeros((w.shape[0], w.shape[1], 2 * Nh, w.shape[3]))
        for cla in range(w.shape[0]):
            P = FN.GLST(w[cla], rfs, Nh, config.list_freqs[cla], config.list_phase[cla])
            for blo in range(w.shape[1]):
                lst[cla, blo:, :, :] = P @ w[cla, blo]                        # nguyen van "LST_transform[cla, blo :, :] = ..."
        trd.append(fbd); trw.append(w); trl.append(lst); trlong.append(L)
    data1c = np.concatenate(trd, axis=1); data1c_raw = np.concatenate(trw, axis=1); data1c_lst = np.concatenate(trl, axis=1); data1c_long = np.concatenate(trlong, axis=1)
    nb = nBlock * nsrc
    train_data = np.zeros((NT * nb, config.Nm, data1c.shape[3], data1c.shape[4])); train_label = np.zeros(NT * nb)
    reference = np.zeros((NT * nb, 2 * Nh, data1c.shape[4])); train_data_lst = np.zeros((NT * nb, 2 * Nh, data1c_lst.shape[3]))
    train_data_raw = np.zeros((NT * nb, data1c.shape[3], data1c.shape[4])); train_long = np.zeros((NT * nb, data1c_long.shape[2], data1c_long.shape[3]))
    for j in range(NT):                                                       # main.py 203-208
        sl = slice(nb * j, nb * j + nb)
        train_data[sl] = data1c[j]; train_data_raw[sl] = data1c_raw[j]; train_label[sl] = j; train_data_lst[sl] = data1c_lst[j]
        reference[sl] = FN.Ref(data1c_raw[j], rfs, Nh, config.list_freqs[j], config.list_phase[j]); train_long[sl] = data1c_long[j]
    train_data_lst, Lst_weights, data_mean = FN.get_LST_transform(train_data_raw, train_data_lst, train_label, list_freqs=config.list_freqs, fs=config.rfs, num_harms=4)
    train_data = fb_stand_f32(train_data)                                     # = FB_stand(...).astype(float32); mang nhan FloatTensor nen gia tri vao mang khong doi
    train_data_lst = fb_stand_f32(train_data_lst)
    for nm_, v_ in (("train_data", train_data), ("train_data_lst", train_data_lst)):
        if not np.isfinite(v_).all():
            raise SystemExit(f"fold {f}: {nm_} khong huu han sau FB_stand")
    dat_seed(seed)
    um = Mm.universal_model()
    model = um.train(train_data, train_data_lst, reference.astype(np.float32), train_label, a.vong or config.n_epochs)
    del train_data, train_data_lst
    # ---- thu tren nguoi f (main.py 226-295) ----
    w, L = D[f]
    data1c_test = filter_bank(L)
    test_new = np.zeros((w.shape[0], w.shape[1], NT, 2 * Nh, w.shape[3]))
    for cla in range(NT):
        P = Lst_weights[cla]
        for idx in range(w.shape[0]):
            for blo in range(w.shape[1]):
                test_new[idx, blo, cla] = P @ w[idx, blo]
    wp = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]; ws = [(3, 92), (12, 92), (20, 92), (28, 92), (36, 92)]
    filterbank = FN.generate_filterbank(wp, ws, srate=rfs, order=15, rp=0.5)
    filterweights = [(i_ + 1) ** (-1.25) + 0.25 for i_ in range(5)]
    Yf = FN.generate_cca_references(config.list_freqs, srate=rfs, T=(config.sample_length if a.che == "goc" else T / rfs), phases=config.list_phase, n_harmonics=5)
    assert Yf.shape[-1] == T
    Xfit = train_data_raw if a.che == "goc" else train_long
    e1 = CCm.FBTtCCA(filterbank=filterbank, n_components=1, filterweights=np.array(filterweights), n_jobs=-1).fit(Xfit, train_label, Yf=Yf)
    e2 = CCm.FBTtCCA(filterbank=filterbank, n_components=1, filterweights=np.array(filterweights), n_jobs=-1).fit(Xfit, train_label, Yf=data_mean)
    data_test = np.zeros((NT * nBlock,) + data1c_test.shape[2:]); data_test_raw = np.zeros((NT * nBlock,) + w.shape[2:])
    data_test_long = np.zeros((NT * nBlock,) + L.shape[2:]); test_label = np.zeros(NT * nBlock); data_test_lst = np.zeros((NT * nBlock, NT, 2 * Nh, w.shape[3]))
    for j in range(NT):
        sl = slice(nBlock * j, nBlock * j + nBlock)
        data_test[sl] = data1c_test[j]; data_test_raw[sl] = w[j]; test_label[sl] = j; data_test_lst[sl] = test_new[j]; data_test_long[sl] = L[j]
    data_test = FN.FB_stand(data_test); data_test_lst = FN.FB_stand(data_test_lst)
    prediction = Mm.universal_model().test(model, data_test, data_test_lst)
    Xte = data_test_raw if a.che == "goc" else data_test_long
    _, f4 = e1.predict(Xte); p2, f2 = e2.predict(Xte)
    _, f1 = FBm.fbcca(Xte, y_ref=FBm.cca_reference(config.list_freqs, config.rfs, config.T, config.Nh))
    p3, f3 = FBm.fbcca(Xte, y_ref=data_mean)
    for nm_, v_ in (("CNN", prediction), ("f1_FBCCA", f1), ("f2_FBTTCCA_new", f2), ("f3_FBCCA_new", f3), ("f4_FBTTCCA", f4)):
        if not np.isfinite(np.asarray(v_, float)).all():                     # diem khong huu han -> dung, khong de argmax am tham tra lop 0
            raise SystemExit(f"fold {f}: diem {nm_} khong huu han o luot {np.where(~np.isfinite(np.asarray(v_, float)).all(1))[0][:10].tolist()}")
    hang = {nm_: int((np.ptp(np.asarray(v_, float), axis=1) == 0).sum()) for nm_, v_ in (("CNN", prediction), ("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4))}
    mm = lambda v: (v - v.min(1, keepdims=True)) / (v.max(1, keepdims=True) - v.min(1, keepdims=True))   # nguyen van ma goc (vecto hang -> NaN, ghi so luot o 'luot_vecto_hang')
    f1, f2, f3, f4, pr = mm(f1), mm(f2), mm(f3), mm(f4), mm(prediction)
    nhan = {"final": np.argmax(pr + (f1 + f2 + f3) / 3, -1), "CNN": np.argmax(prediction, -1), "FBTTCCA_new": np.asarray(p2), "CCA_new": np.asarray(p3),
            "FBTTCCA_new+CCA_new": np.argmax(f2 + f3, -1), "CNN+FBCCA": np.argmax(pr + f1, -1), "CNN+FBTTCCA_new": np.argmax(pr + f2, -1),
            "CNN+FBCCA_new": np.argmax(pr + f3, -1), "CNN+FBTTCCA_new+FBCCA_new": np.argmax(pr + (f3 + f2) / 2, -1),
            "CNN+FBTTCCA_new+FBCCA": np.argmax(pr + (f1 + f2) / 2, -1), "CNN+FBTTCCA": np.argmax(pr + f4, -1)}
    blk = np.tile(np.arange(nBlock), NT); cls = np.repeat(np.arange(NT), nBlock)
    assert np.array_equal(cls, test_label.astype(int))
    luot = blk * NT + cls                                                     # chi so luot theo thu tu khoi roi lop (0-based)
    od = np.argsort(luot); y = cls[od]
    e = dict(fold=f, seed=seed, che=a.che, ds=a.ds, T=a.T, ns=T, du_nhanh=a.du_nhanh, thoi_gian_s=time.time() - t0, luot_vecto_hang=hang)
    for k, v in nhan.items():
        d_ = np.asarray(v).astype(int)[od]
        e[k] = dict(moi_luot=float((d_ == y).mean()), sau=float((d_[40:] == y[40:]).mean()))
    e["du_doan_final"] = nhan["final"].astype(int)[od].tolist(); e["du_doan_CNN"] = nhan["CNN"].astype(int)[od].tolist(); e["y"] = y.tolist()
    return e


def main():
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(ds=a.ds, T=a.T, che=a.che, du_nhanh=a.du_nhanh, vong=a.vong or config.n_epochs, ns=T, rfs=rfs, NT=NT, Nh=Nh, seed=(41 if a.che == "goc" else SEED),
               bat_dau_cua_so=(c["ld_goc"] if a.che == "goc" else st_gt), script_sha256=sha(os.path.abspath(__file__)), shim_sha256=sha(os.path.join(SHIM, "config.py")),
               goc_sha256={l.split()[1].lstrip("*"): l.split()[0] for l in open(os.path.join(GOCD, "SHA256SUMS.txt"))}, commit="1660d42113d5721c5ec05d5f8b92aeabcd50c31f",
               list_freqs=[float(v) for v in config.list_freqs], list_phase_pi=[float(v / np.pi) for v in config.list_phase], kenh=c["ch"], n_nguoi=c["n"],
               du_lieu=c["root"], phien_ban={m_.__name__: getattr(m_, "__version__", "") for m_ in (np, torch, sys.modules["scipy"], sys.modules["sklearn"])},
               ghi_chu_seed=("seed 41 dat lai dau MOI fold (ma goc dat mot lan truoc vong LOSO, RNG chay tiep qua cac fold): bien the seed theo fold"
                             if a.che == "goc" else "seed 20261107 + 1000 fold + round(10 T)"))
    cf = os.path.join(a.out, "config.json")
    if os.path.exists(cf):
        if json.load(open(cf)) != cfg:
            sys.exit("config khac lan chay truoc: khong tron ket qua")
    else:
        tmp = f"{cf}.{os.getpid()}.tmp"; json.dump(cfg, open(tmp, "w"), indent=1); os.replace(tmp, cf)
    k, m = [int(v) for v in a.folds.split("::")]
    print(f"nap {c['n']} nguoi ({a.ds}), T = {a.T}, che {a.che}, du_nhanh {a.du_nhanh}, cuda {torch.cuda.is_available()}", flush=True)
    D = {s: nap(s) for s in range(1, c["n"] + 1)}
    for f in range(k, c["n"] + 1, m):
        fn = os.path.join(a.out, f"fold{f:03d}.json")
        if os.path.exists(fn):
            continue
        e = chay_fold(f, D); e.update(script_sha256=cfg["script_sha256"])
        tmp = fn + ".tmp"; json.dump(e, open(tmp, "w")); os.replace(tmp, fn)
        print(f"fold {f}: xong {e['thoi_gian_s']:.0f} s", flush=True)
    import resource
    print("RSS_DINH_KB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)    # nut tinh khong co /usr/bin/time


main()
