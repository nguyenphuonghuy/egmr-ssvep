#!/usr/bin/env python
"""Doi thu SFDA (Guney va cong su, J. Neural Eng. 2025, DOI 10.1088/1741-2552/ae0c3d) cho nhanh b1gop, diem 6
Chuyen tu ma MATLAB cong khai osmanberke/SFDA-SSVEP-BCI (GPL-3.0), ban luu o
tham-chieu-matlab/: SFDA_main.m, ourClassificationLayer.m, neighbors_calculation.m, silhouette_score_calculation.m.
Mang nen = Global DNN thu nho cua dung fold (W0, chung voi E0, E4). Tam sua doi:
 (i) 22 bo loc l2 thay hang 120; (ii) kien truc thu nho, dropout hai lop dau 0,6 (Benchmark) / 0,7 (BETA) khi hoc; (iii) PyTorch, SGDM momentum 0,9,
 lr 1e-4, L2 1e-3 chi tren trong so, toan lo, 50 buoc moi vong; (iv) khong tran so vong; (v) vong tim do sut dung o cot N - 2 (ma goc loi chi so);
 (vi) tap loc silhouette rong: so hang tu thich nghi = 0; (vii) giu mau gia: khi nhan gia thieu lop, mau so cua so hang cuc bo la N + N_lop trong
 MOI lan thu khi iter == 1, bo sau lan nhan dau (mau gia co dau vao 0, khong vao mat na, khong co lang gieng: chi anh huong mau so);
 (viii) ban truc tuyen: moc khong co vong nao duoc nhan thi dung mang cua moc nhan gan nhat, chua co thi W0.
Quy uoc silhouette (ham silhouette cua MATLAB, khoang cach cosine): diem cua cum mot phan tu = 1; chi co mot cum: moi diem = NaN nhu MATLAB:
tong diem NaN, mat na (>= 0) toan sai, moi phep so sanh voi NaN deu sai (khong chon FBCCA, khong nhan vong).
Chon lam (ngoai main.m; quy tac cua ban chuyen): lam co tong silhouette lon nhat, bo qua NaN; moi lam deu NaN thi lay lam dau va coi nhu khong co mang.
Che do: --che-do lo (lop (b): theo lo, chuyen nap, cham tren chinh du lieu do) | tt (lop (c): moc a = 40, 60, ..., N - 20; giai 20 luot ke tiep).
Nhan that chi de cham.
"""
import argparse, copy, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 20261107
N0, LO, NCH = 40, 20, 22
DELTA, B_THU, J_BUOC = 0.05, 3, 50
LAMS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
P2 = {"benchmark": 0.6, "beta": 0.7}


def gop_dai(model, X):
    return (model.l1.weight.view(1, -1, 1, 1) * X).sum(1)                       # (L, C, ns)


def sil_cosine(D, lab):
    """D (L, ns) da tru TB hang; lab (L,) long. Diem silhouette tung mau, khoang cach cosine, quy uoc MATLAB."""
    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-30)
    dist = 1.0 - Dn @ Dn.T
    cls, inv = torch.unique(lab, return_inverse=True)
    K = len(cls); L = len(lab)
    if K == 1:
        return torch.full((L,), float("nan"), dtype=D.dtype, device=D.device)
    M = torch.zeros(L, K, dtype=D.dtype, device=D.device); M[torch.arange(L), inv] = 1.0
    cnt = M.sum(0)
    tong = dist @ M                                                             # (L, K) tong khoang cach toi tung cum
    n_own = cnt[inv]
    a = tong[torch.arange(L), inv] / (n_own - 1).clamp_min(1.0)
    tb = tong / cnt[None, :]
    tb[torch.arange(L), inv] = float("inf")
    b = tb.min(1).values
    s = (b - a) / torch.maximum(a, b).clamp_min(1e-30)
    return torch.where(n_own > 1, s, torch.ones_like(s))


def silhouette(model, X, lab):
    """Nhu silhouette_score_calculation.m: thu tung bo loc l2 (to hop kenh), chon bo co TONG silhouette lon nhat.
    Tra (to hop kenh (C,), tong diem, diem tung mau, mat na diem >= 0)."""
    sub = gop_dai(model, X)
    W = model.l2.weight[:, 0, :, 0]                                             # (nch, C)
    tot, vals = [], []
    for j in range(W.shape[0]):
        P = torch.einsum("c,lcn->ln", W[j], sub)
        P = P - P.mean(1, keepdim=True)
        v = sil_cosine(P, lab); vals.append(v); tot.append(v.sum())
    tot = torch.stack(tot)
    j = int(torch.argmax(tot))                                                  # sort giam dan, lay dau: phan tu lon nhat dau tien
    return W[j], float(tot[j]), vals[j], vals[j] >= 0


def lang_gieng(model, X, lab, lam, delta=DELTA):
    """Nhu neighbors_calculation.m. Tra (tong diem silhouette, ind (L,) bool, cap (i, j, w) cho so hang cuc bo)."""
    comb, score, _, ind = silhouette(model, X, lab)
    P = torch.einsum("c,lcn->ln", comb, gop_dai(model, X))
    R = torch.corrcoef(P)
    sc, si = torch.sort(R, dim=1, descending=True)
    L = R.shape[0]
    diffs = (sc[:, 1:L - 1] - sc[:, 2:L]) / sc[:, 1:L - 1]                      # (L, L - 2)
    nho = (diffs < delta).to(torch.int64)
    k = 1 + torch.cumprod(nho, dim=1).sum(1)                                    # sua (v): dung o cot L - 2, k toi da L - 1
    si = si.cpu().numpy(); k = k.cpu().numpy(); indn = ind.cpu().numpy()
    pi, pj, pw = [], [], []
    for i in range(L):
        nn = [int(j) for j in si[i, 1:k[i] + 1] if indn[j]]
        if not nn and lam != 0 and indn[i]:
            nn = [int(si[i, 0])]
        if nn:
            w = ((1 - lam) if (indn[i] or lam == 1) else 1.0) / len(nn)
            pi += [i] * len(nn); pj += nn; pw += [w] * len(nn)
    dev = X.device
    return score, ind, (torch.tensor(pi, dtype=torch.long, device=dev), torch.tensor(pj, dtype=torch.long, device=dev),
                        torch.tensor(pw, dtype=X.dtype, device=dev))


def mat_mat(logits, T, ind, cap, lam, mau_so):
    """Nhu ourClassificationLayer.m: lam * CE_tb(tap ind) + (tong w * CE(y_i, nhan cua lang gieng j)) / mau_so. Tra (tong, sl, ll)."""
    lp = Fn.log_softmax(logits, dim=1)
    sl = -lp[ind, T[ind]].mean() if bool(ind.any()) else lp.sum() * 0.0        # sua (vi)
    pi, pj, pw = cap
    ll = -(pw * lp[pi, T[pj]]).sum() / mau_so if len(pi) else lp.sum() * 0.0
    return lam * sl + ll, sl, ll


def du_doan(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).argmax(1)


def hoc_vong(model, sd, X, T, ind, cap, lam, mau_so, p2):
    model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad_(True)
    model.d1.p = p2; model.d2.p = p2
    ps = list(model.named_parameters())
    opt = torch.optim.SGD([{"params": [p for n, p in ps if n.endswith("weight")], "weight_decay": 1e-3},
                           {"params": [p for n, p in ps if n.endswith("bias")], "weight_decay": 0.0}], lr=1e-4, momentum=0.9)
    model.train()
    for _ in range(J_BUOC):
        opt.zero_grad(set_to_none=True)
        mat_mat(model(X), T, ind, cap, lam, mau_so)[0].backward(); opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return copy.deepcopy(model.state_dict())


def sfda(model, sd0, X, fb, lam, p2, n_lop):
    """Mot gia tri lam, nhu vong while cua SFDA_main.m. Tra dict: sd (None neu khong vong nao duoc nhan), Y (nhan cuoi), score, so_lan, nhanh, gia, mau_so."""
    model.load_state_dict(sd0)
    Y = du_doan(model, X)
    score, ind, cap = lang_gieng(model, X, Y, lam)
    nhanh = "mang"
    if silhouette(model, X, fb)[1] > score:
        Y = fb.clone(); score, ind, cap = lang_gieng(model, X, Y, lam); nhanh = "fbcca"
    gia = int(len(torch.unique(Y))) != n_lop
    it, thu, pre, sd_cur, so_lan, ms = 1, 0, score, sd0, 0, []
    while thu < B_THU:
        so_lan += 1
        mau_so = X.shape[0] + (n_lop if (gia and it == 1) else 0)              # sua (vii)
        ms.append(mau_so)
        sd_new = hoc_vong(model, sd_cur, X, Y, ind, cap, lam, mau_so, p2)
        Yn = du_doan(model, X)
        sc = silhouette(model, X, Yn)[1]
        if sc > pre:
            it += 1; pre = sc; thu = 0; Y = Yn; sd_cur = sd_new
            _, ind, cap = lang_gieng(model, X, Y, lam)
        else:
            thu += 1
    if it != 1:
        model.load_state_dict(sd_cur); Y = du_doan(model, X); score = silhouette(model, X, Y)[1]
        return dict(sd=sd_cur, Y=Y, score=score, so_lan=so_lan, nhanh=nhanh, gia=gia, mau_so=ms)
    model.load_state_dict(sd0)
    return dict(sd=None, Y=Y, score=silhouette(model, X, Y)[1], so_lan=so_lan, nhanh=nhanh, gia=gia, mau_so=ms)


def sfda_chon(model, sd0, X, fb, p2, n_lop):
    """Chay moi lam, chon lam co tong silhouette lon nhat (lam dau tien khi hoa)."""
    kq = [sfda(model, sd0, X, fb, lam, p2, n_lop) for lam in LAMS]
    diem = np.array([k["score"] for k in kq], float)
    j = int(np.nanargmax(diem)) if np.isfinite(diem).any() else 0
    if not np.isfinite(diem[j]):
        kq[j]["sd"] = None
    so = lambda v: (float(v) if np.isfinite(v) else None)
    tt = dict(lam=LAMS[j], score=so(kq[j]["score"]), nhanh=kq[j]["nhanh"], gia=bool(kq[j]["gia"]), co_mang=kq[j]["sd"] is not None,
              so_lan=[k["so_lan"] for k in kq], diem=[so(k["score"]) for k in kq])
    return kq[j], tt


def truc_tuyen(model, sd0, X, fb, p2, n_lop, d0):
    """Ban truc tuyen (lop (c)): moc a = N0, N0 + LO, ..., L - LO; SFDA tren luot 1..a tu W0; giai LO luot ke tiep bang mang cua moc nhan gan nhat
    (chua co thi W0: quy tac (viii)). d0: quyet dinh cua W0 (numpy). Tra (dec, danh sach thong tin moc)."""
    L = X.shape[0]; dec = d0.copy(); cur = sd0; moc = []
    for m in range(N0, L, LO):
        kq, tt = sfda_chon(model, sd0, X[:m], fb[:m], p2, n_lop)
        tt["a"] = m; tt["du_phong"] = kq["sd"] is None
        if kq["sd"] is not None:
            cur = kq["sd"]
        model.load_state_dict(cur)
        dec[m:m + LO] = du_doan(model, X[m:m + LO]).cpu().numpy()
        moc.append(tt)
    return dec, moc


def fbcca(Xraw, ds, freqs, ns, B):
    """FBCCA nhan qua theo tien xu ly cua du an (5 dai, 5 hai, tong r^2 co trong so (k^-1,25 + 0,25)), nhu fbcca_classification.m."""
    c = B.DS[ds]; fs = c["fs"]
    P = B.preprocess(Xraw, ds, B.filterbank(fs), ns)                             # (L, 5, C, ns)
    Qy = B.refs(freqs, fs, ns)                                                   # (N, ns, 2H)
    Qy2 = Qy.transpose(1, 0, 2).reshape(ns, -1)
    out = np.empty(P.shape[0], int)
    for l in range(P.shape[0]):
        d = np.zeros(len(freqs))
        for k in range(P.shape[1]):
            x = P[l, k] - P[l, k].mean(1, keepdims=True)
            Qx = np.linalg.qr(x.T)[0]
            M = (Qx.T @ Qy2).reshape(Qx.shape[1], len(freqs), -1).transpose(1, 0, 2)
            d += B.FBW[k] * np.linalg.svd(M, compute_uv=False)[:, 0] ** 2
        out[l] = int(np.argmax(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta"])
    ap.add_argument("--che-do", required=True, choices=["lo", "tt"])
    ap.add_argument("--T", type=float, required=True, choices=[0.6, 1.0])
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nguoi", default="all")
    ap.add_argument("--thiet-bi", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    if a.thiet_bi == "cuda" and not torch.cuda.is_available():
        sys.exit("khong co CUDA")
    dev = torch.device(a.thiet_bi)
    print("thiet bi:", dev, flush=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    import b1_diem as B, global_dnn as GD
    for d in B.DS:
        B.DS[d]["root"] = os.environ.get("B1_ROOT_" + d.upper(), B.DS[d]["root"])
    ds = a.ds; c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]; n_lop = GD.NT
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))
    assert cfg["nch"] == NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg["subjects"] == 0 and cfg.get("ds", "benchmark") == ds
    ng = list(range(1, nsub + 1)) if a.nguoi == "all" else list(range(int(a.nguoi.split("-")[0]), int(a.nguoi.split("-")[1]) + 1))
    t0 = time.time(); per = []
    for s in ng:
        seed = SEED + 1000 * s + int(round(a.T * 10))
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        Xr, y, freqs = B.load_subject(ds, s, B.CH9)
        nl = {"benchmark": 240, "beta": 160}[ds]
        assert Xr.shape[0] == nl and Xr.shape[1] == 9 and np.isfinite(Xr).all() and np.array_equal(np.bincount(y, minlength=n_lop), np.full(n_lop, nl // n_lop)), f"du lieu nguoi {s}"
        X = torch.tensor(GD.tien_xu_ly(Xr, ds, "causal", ns, fs), device=dev)
        fb = torch.tensor(fbcca(Xr, ds, freqs, ns, B), device=dev, dtype=torch.long)
        model = GD.GlobalDNN(X.shape[2], ns, nch=NCH, p_cuoi=0.95).to(dev)
        ftr = os.path.join(a.trongso, f"fold{s:03d}_trongso.pt")
        sd0 = torch.load(ftr, map_location=dev)
        model.load_state_dict(sd0); sd0 = copy.deepcopy(model.state_dict())
        for p in model.parameters():
            p.requires_grad_(False)
        d0 = du_doan(model, X).cpu().numpy()
        L = len(y); e = {"fold": s, "seed": seed, "trongso_sha256": hashlib.sha256(open(ftr, "rb").read()).hexdigest(), "tinh": dict(sau=float((d0[N0:] == y[N0:]).mean()), phien=float((d0 == y).mean())),
                         "fbcca_phien": float((fb.cpu().numpy() == y).mean())}
        if a.che_do == "lo":
            kq, tt = sfda_chon(model, sd0, X, fb, P2[ds], n_lop)
            Yc = kq["Y"].cpu().numpy()
            e["sfda"] = dict(phien=float((Yc == y).mean()), dec=[int(v) for v in Yc]); e["moc"] = [tt]
        else:
            dec, moc = truc_tuyen(model, sd0, X, fb, P2[ds], n_lop, d0)
            e["sfda"] = dict(sau=float((dec[N0:] == y[N0:]).mean()), phien=float((dec == y).mean()), dec=[int(v) for v in dec]); e["moc"] = moc
        per.append(e)
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"nguoi {s} xong {time.time()-t0:.0f} s", flush=True)              # khong in do chinh xac
    mac = GD.dem_mac(None, 9, ns, nch=NCH)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    js = dict(seed=SEED, ds=ds, che_do=a.che_do, T=a.T, n0=N0, lo=LO, nch=NCH, delta=DELTA, b_thu=B_THU, j_buoc=J_BUOC, lams=LAMS, p2=P2[ds], thiet_bi=str(dev),
              nguoi=ng, trongso=a.trongso, trongso_script_sha256=cfg["script_sha256"], mac_suy_luan=mac["tong"],
              mac_hoc_moi_mau_moi_buoc=3 * mac["tong"] - mac["l1"],
              mac_ghi_chu="du toan phep nhan chinh; phan hoc = 50 buoc x so vong x so mau x mac_hoc_moi_mau_moi_buoc; KHONG gom luot xuoi kiem nhan vong, chon nhan dau, FBCCA, silhouette, lang gieng",
              script_sha256=sha(os.path.abspath(__file__)), global_dnn_sha256=sha(GD.__file__), b1_diem_sha256=sha(B.__file__), tung_nguoi=per)
    json.dump(js, open(a.out, "w"), indent=1)
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
