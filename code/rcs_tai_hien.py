#!/usr/bin/env python
"""TAI HIEN RCS (Shi va cong su, TNSRE 2023, DOI 10.1109/TNSRE.2023.3245654) tren giao thuc phat lai cua kh_deu, de kiem H3-b.
LOSO, thu tu luot = thu tu tep (khoi roi muc tieu).

Theo toan van (muc II-A..II-C):
  - Mo hinh nguon: TRCA gop lop (ensemble) co dai loc + FBCCA, trong so 1:1 ("TRCA w/R"); he so cong qua dai voi a_b = b^-1.25 + 0.25, dung rho (khong binh phuong) nhu cong thuc bai goc.
  - Do tin: alpha = kiem dinh t mot phia giua rho_q (lon nhat) va cac rho con lai (phuong sai gop, bac tu do N - 2).
  - Gioi thieu: xep hang alpha tich luy (TB qua luot) cua tung mo hinh nguon; dung khi tap K_REP = 5 nguon dung dau khong doi trong M_ON luot lien tiep;
    muon nhat sau N_REC = 20 luot. Trong luc gioi thieu, quyet dinh bang FBCCA.
  - Sau gioi thieu: TRCA w/R hoc tren du lieu gop cua 5 dai dien.
  - Thich nghi theo lop bang nhan yeu (nhan = quyet dinh da phat): lop co < 2 luot nguoi dung: noi thang; >= 2: LST tung luot dai dien
    (P = Xtb chi^T (chi chi^T)^-1, Xtb = TB cac luot nhan yeu cua lop) roi noi; hoc lai bo loc va mau cua cac lop co thay doi.
Lua chon o cho bai goc khong ghi: M_ON = 5; cua so co dinh (khong co cua so theo lo); nhip hoc lai CHINH = sau moi lo 20 luot, PHU = sau moi luot;
  5 dai, 5 hai, 9 kenh va bo loc nhan qua cua du an.
Bien the:
  day_du_lo / day_du_luot : 5 dai TRCA + FBCCA 5 dai (khong gioi han MAC).
  tran_lo / tran_luot     : "RCS trong tran" (H3-b): quyet dinh <= 1 M MAC: chi TRCA gop lop (bo so hang FBCCA vi rieng FBCCA mot dai da gan 1 M o 1,0 s),
                            so dai lon nhat sao cho nb * Ns * (C*N + N*N + N) <= 1 M; trong luc gioi thieu quyet dinh bang FBCCA 1 dai.
                            Phan tinh alpha cua moi nguon luc gioi thieu KHONG bi gioi han (co loi cho RCS), bao rieng.
Moi luot: giai bang mo hinh hien co -> ghi quyet dinh -> moi dua vao thich nghi. Cham: luot N0+1..het ("sau") va ca phien.
Kiem tai hien: mo hinh 5 dai dien (khong thich nghi) cham tren moi luot bang nhan that (rep_wR, rep_trca); bai goc: 0,79 +- 0,18 (TRCA, 1 s, 55 nguoi).
Sua 2026-11-14: (1) bien the trong tran chi TINH so dai duoc dung khi cham diem; (2) --chi-cong: chi tinh cong tai hien
(rep_wR, Benchmark, 1,0 s), ma thoat 3 neu ngoai 0,65-0,85; (3) them bien the BAO KEM tran1R_* = 1 dai TRCA + 1 dai FBCCA o cua so ma tong <= 1 M (chi 0,6 s);
(4) lo tinh tu dau phien: neu gioi thieu ket thuc dung moc lo (luot 20) thi hoc lai ngay bang nhan da phat; nhip moi luot hoc lai ngay sau gioi thieu.
Sua them 2026-11-14: tran1R_* = 1 dai TRCA + 1 dai FBCCA voi SO HAI LON NHAT (<= 5) sao cho tong <= 1 M (0,6 s: 5 hai; 0,8 s: 3; 1,0 s: 2),
de doi thu trong tran van giu thanh phan tham chieu sin/cos o MOI cua so. Tuong quan 2 chieu: cac hang cua W^T x va W^T mau deu co TB 0 theo thoi gian
(x va mau da tru TB theo thoi gian), nen cosin = Pearson 2 chieu (khong can tru TB lai).
--im-lang: chay thong, khong in so, khong ghi tep.
Them 2026-11-15: tranG_* = TRCA gop lop DU nbc dai + FBCCA kF dai (5 hai) voi kF lon nhat sao cho tong <= 1 M; chi chay khi kF >= 1
(40 lop: kF = 0 nen khong co; 9 lop: kF = 3). Muc dich: khong bo sot ban RCS manh nhat con vua tran khi so lop nho.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
from scipy.linalg import eigh
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B  # noqa: E402
import du_lieu_eld as DE  # noqa: E402

SEED = 20261107
WINS = [0.6, 0.8, 1.0]
N0, K_REP, M_ON, N_REC, LO = 40, 5, 5, 20, 20
NB, NT = B.NB, B.NTGT
REG = 1e-6
CAP = 1000000
A = np.asarray(B.FBW, float)


def top_gev(S, Q):
    C = S.shape[0]
    ev, V = eigh((S + S.T) / 2, Q + REG * np.trace(Q) / C * np.eye(C))
    return V[:, -1]


def trca_w(D):
    """D: (m, NB, C, ns) cac luot cua mot lop. Tra (NB, C)."""
    s = D.sum(0)
    Q = np.einsum("mbct,mbdt->bcd", D, D)
    S = np.einsum("bct,bdt->bcd", s, s) - Q
    return np.stack([top_gev(S[b], Q[b]) for b in range(NB)])


def chieu_mau(W, Mt):
    """W (NB, C, N), Mt (N, NB, C, ns) -> Tp (NB, N, N, ns) = mau lop k chieu qua bo loc gop, va chuan (NB, N)."""
    Tp = np.einsum("bcf,kbct->bkft", W, Mt)
    return Tp, np.sqrt((Tp ** 2).sum((2, 3)))


def diem(W, Tp, Tn, x):
    """x (NB, C, ns) -> r (NB, N): tuong quan 2 chieu giua W^T x va W^T mau_k."""
    z = np.einsum("bcf,bct->bft", W, x)
    num = np.einsum("bft,bkft->bk", z, Tp)
    return num / (np.sqrt((z ** 2).sum((1, 2)))[:, None] * Tn + 1e-300)


def alpha_t(rho):
    q = int(rho.argmax()); rest = np.delete(rho, q); n2 = rest.size
    tt = (rho[q] - rest.mean()) / (rest.std(ddof=1) * np.sqrt(1.0 + 1.0 / n2) + 1e-300)
    return float(tdist.sf(tt, n2 - 1))                                      # phuong sai gop voi n1 = 1: bac tu do = n2 - 1 = N - 2


def lst(chi, xtb):
    """chi (m, NB, C, ns) luot dai dien; xtb (NB, C, ns). Tra chi' = P chi voi P = xtb chi^T (chi chi^T)^-1 theo tung luot, tung dai."""
    C = chi.shape[2]
    G = np.einsum("mbct,mbdt->mbcd", chi, chi)
    G = G + REG * (np.trace(G, axis1=2, axis2=3) / C)[..., None, None] * np.eye(C)
    XC = np.einsum("bct,mbdt->mbcd", xtb, chi)
    P = np.swapaxes(np.linalg.solve(np.swapaxes(G, 2, 3), np.swapaxes(XC, 2, 3)), 2, 3)   # P G = XC
    return np.einsum("mbcd,mbdt->mbct", P, chi)


def refs_h(freqs, fs, ns, h):
    tt = np.arange(ns) / fs
    Yr = np.stack([np.concatenate([[np.sin(2 * np.pi * m * f * tt), np.cos(2 * np.pi * m * f * tt)] for m in range(1, h + 1)]) for f in freqs])
    Yr = Yr - Yr.mean(-1, keepdims=True)
    return np.stack([np.linalg.qr(v.T)[0] for v in Yr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["benchmark", "beta", "eld", "dong"])
    ap.add_argument("--im-lang", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--chi-cong", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    global NT
    if a.ds == "eld":                                                      # them 2026-11-15: eldBETA 9 muc tieu (H3 tren bo chua xem voi RCS)
        import functools
        DE.dang_ky_B(B); NT = 9; B.macs = functools.partial(B.macs, n=NT)
    if a.ds == "dong":                                                     # them 2026-11-17: Dong 2023 (40 muc tieu, 8 kenh; bao kem)
        import du_lieu_dong as DG
        DG.dang_ky(B, None, "dong")
    c = B.DS[a.ds]; fs = c["fs"]; sos = B.filterbank(fs)
    nsub = a.subjects or c["n"]
    ns_max = int(round(max(WINS) * fs))
    t0 = time.time()
    P, Y, freqs = [], [], None
    for s in range(1, nsub + 1):
        X, y, fr = B.load_subject(a.ds, s, B.CH9)
        if freqs is None:
            freqs = np.asarray(fr, float)
        assert np.allclose(freqs, fr)
        P.append(B.preprocess(X, a.ds, sos, ns_max).astype(np.float32)); Y.append(y)
    C = P[0].shape[2]
    print(f"nap {nsub} nguoi: {time.time()-t0:.0f} s", flush=True)
    res = {}
    for T in ([1.0] if a.chi_cong else WINS):
        ns = int(round(T * fs))
        u_trca = ns * (C * NT + NT * NT + NT)                                 # MAC mot dai TRCA gop lop: chieu + N tuong quan + chuan
        nbc = int(min(NB, CAP // u_trca))
        assert nbc >= 1
        mac_fb1 = B.macs("fbcca", C, ns)
        hR = max([h for h in range(1, B.H + 1) if u_trca + B.macs("fbcca", C, ns, h2=2 * h) <= CAP] or [0])
        co_1R = hR >= 1
        kF = max([k for k in range(0, NB + 1) if nbc * u_trca + k * mac_fb1 <= CAP]) if nbc == NB else 0
        QrR = refs_h(freqs, fs, ns, hR) if co_1R else None
        X = [B.center(p[..., :ns].astype(np.float64)) for p in P]
        Qr = B.refs(freqs, fs, ns)
        # FBCCA tung dai, rho (khong binh phuong)
        FB = []; FBR = []
        for x in X:
            f = np.empty((x.shape[0], NB, NT))
            for i in range(x.shape[0]):
                for b in range(NB):
                    qx = np.linalg.qr(x[i, b].T)[0]
                    f[i, b] = np.linalg.svd(np.einsum("tc,kth->kch", qx, Qr), compute_uv=False)[:, 0]
            FB.append(f)
            if co_1R:                                                         # FBCCA 1 dai, hR hai (cho tran1R)
                g = np.empty((x.shape[0], NT))
                for i in range(x.shape[0]):
                    qx = np.linalg.qr(x[i, 0].T)[0]
                    g[i] = np.linalg.svd(np.einsum("tc,kth->kch", qx, QrR), compute_uv=False)[:, 0] * A[0]
                FBR.append(g)
        # mo hinh TRCA gop lop cua tung nguoi (dung khi nguoi do la NGUON; nguoi thu u khong bao gio dung mo hinh cua chinh minh)
        Wsrc, Tsrc, Nsrc = [], [], []
        for s, x in enumerate(X):
            W = np.empty((NB, C, NT)); Mt = np.empty((NT, NB, C, ns))
            for k in range(NT):
                D = x[Y[s] == k]
                W[:, :, k] = trca_w(D); Mt[k] = D.mean(0)
            Tp, Tn = chieu_mau(W, Mt)
            Wsrc.append(W); Tsrc.append(Tp.astype(np.float32)); Nsrc.append(Tn)
        print(T, f"mo hinh nguon xong {time.time()-t0:.0f} s; so dai trong tran = {nbc}", flush=True)
        per = []
        for u in range(nsub):
            src = [s for s in range(nsub) if s != u]
            x = X[u]; y = Y[u]; n = len(y); fb = FB[u]
            fb5 = np.einsum("ibk,b->ik", fb, A)                               # FBCCA 5 dai
            fb1 = fb[:, 0] * A[0]                                             # FBCCA 1 dai
            # ---- gioi thieu dai dien (khong dung nhan) ----
            acc = np.zeros(len(src)); prev = None; cnt = 0; rec_end = N_REC; top = None
            for i in range(N_REC):
                for j, s in enumerate(src):
                    r = diem(Wsrc[s], Tsrc[s], Nsrc[s], x[i])
                    acc[j] += alpha_t(fb5[i] + r.T @ A)
                top = frozenset(np.argsort(acc / (i + 1), kind="stable")[:K_REP].tolist())
                cnt = cnt + 1 if top == prev else 1
                prev = top
                if cnt >= M_ON:
                    rec_end = i + 1
                    break
            reps = [src[j] for j in sorted(top)]
            Drep = [np.concatenate([X[r][Y[r] == k] for r in reps]) for k in range(NT)]
            W0 = np.empty((NB, C, NT)); M0 = np.empty((NT, NB, C, ns))
            for k in range(NT):
                W0[:, :, k] = trca_w(Drep[k]); M0[k] = Drep[k].mean(0)
            Tp0, Tn0 = chieu_mau(W0, M0)
            e = {"rec_end": int(rec_end), "dai_dien": [int(r) + 1 for r in reps]}
            # kiem tai hien: mo hinh dai dien khong thich nghi, moi luot, nhan that chi de cham
            r_all = np.stack([diem(W0, Tp0, Tn0, x[i]) for i in range(n)])    # (n, NB, N)
            trca5 = np.einsum("ibk,b->ik", r_all, A)
            e["rep_trca"] = float((trca5.argmax(1) == y).mean()); e["rep_wR"] = float(((trca5 + fb5).argmax(1) == y).mean())
            e["fbcca5"] = float((fb5.argmax(1) == y).mean()); e["fbcca1"] = float((fb1.argmax(1) == y).mean())
            e["fbcca5_sau"] = float((fb5[N0:].argmax(1) == y[N0:]).mean())
            if a.chi_cong:
                per.append(e); continue

            def phat_lai(nb, dung_R, moi_luot, fb_gt, fb_R=None):
                dec = np.empty(n, int)
                dec[:rec_end] = fb_gt[:rec_end].argmax(1)
                W = W0.copy(); Mt = M0.copy(); Tp, Tn = Tp0, Tn0
                ud = [[] for _ in range(NT)]; cho = list(range(rec_end)); so_lan = 0

                def hoc_lai(cho):
                    doi = set()
                    for j in cho:
                        ud[dec[j]].append(j); doi.add(int(dec[j]))
                    for k in doi:
                        U = x[ud[k]]
                        D = np.concatenate([lst(Drep[k], U.mean(0)) if len(ud[k]) >= 2 else Drep[k], U])
                        W[:, :, k] = trca_w(D); Mt[k] = D.mean(0)
                    return chieu_mau(W, Mt)

                if moi_luot or rec_end % LO == 0:                             # lo tinh tu dau phien; nhan = quyet dinh FBCCA da phat
                    Tp, Tn = hoc_lai(cho); cho = []; so_lan += 1
                for i in range(rec_end, n):
                    r = diem(W[:nb], Tp[:nb], Tn[:nb], x[i, :nb])            # chi tinh so dai duoc dung
                    rho = r.T @ A[:nb]
                    if dung_R:
                        rho = rho + (fb5[i] if fb_R is None else fb_R[i])
                    dec[i] = int(rho.argmax())                                # quyet dinh PHAT TRUOC khi luot i vao thich nghi
                    cho.append(i)
                    if moi_luot or (i + 1) % LO == 0:
                        Tp, Tn = hoc_lai(cho); cho = []; so_lan += 1
                return dict(sau=float((dec[N0:] == y[N0:]).mean()), phien=float((dec == y).mean()), so_lan_hoc_lai=so_lan)

            e["day_du_lo"] = phat_lai(NB, True, False, fb5)
            e["day_du_luot"] = phat_lai(NB, True, True, fb5)
            e["tran_lo"] = phat_lai(nbc, False, False, fb1)
            e["tran_luot"] = phat_lai(nbc, False, True, fb1)
            if co_1R:                                                         # bao kem: 1 dai TRCA + 1 dai FBCCA (vua tran)
                e["tran1R_lo"] = phat_lai(1, True, False, fb1, FBR[u]); e["tran1R_luot"] = phat_lai(1, True, True, fb1, FBR[u])
            if kF >= 1:                                                       # TRCA gop lop 5 dai + FBCCA kF dai (vua tran)
                fbk = np.einsum("ibk,b->ik", fb[:, :kF], A[:kF])
                e["tranG_lo"] = phat_lai(NB, True, False, fb1, fbk); e["tranG_luot"] = phat_lai(NB, True, True, fb1, fbk)
            per.append(e)
            if not a.im_lang and (u + 1) % 5 == 0:
                print(T, u + 1, f"{time.time()-t0:.0f} s", flush=True)
        keys = ["rep_trca", "rep_wR", "fbcca5", "fbcca1", "fbcca5_sau", "rec_end"]
        tb = {k: float(np.mean([p[k] for p in per])) for k in keys}
        if a.chi_cong:
            cong = bool(0.65 <= tb["rep_wR"] <= 0.85)
            js = dict(ds=a.ds, T=T, n_subjects=nsub, tb=tb, tung_nguoi=per, khoang=[0.65, 0.85], cong_dat=cong, script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
            json.dump(js, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)      # ghi nguyen tu
            print("CONG TAI HIEN", json.dumps(tb), "DAT" if cong else "KHONG DAT", flush=True)
            sys.exit(0 if cong else 3)
        for v in ["day_du_lo", "day_du_luot", "tran_lo", "tran_luot"] + (["tran1R_lo", "tran1R_luot"] if co_1R else []):
            tb[v + "_sau"] = float(np.mean([p[v]["sau"] for p in per])); tb[v + "_phien"] = float(np.mean([p[v]["phien"] for p in per]))
        for v in (["tranG_lo", "tranG_luot"] if kF >= 1 else []):
            tb[v + "_sau"] = float(np.mean([p[v]["sau"] for p in per])); tb[v + "_phien"] = float(np.mean([p[v]["phien"] for p in per]))
        mac = {"tranG_so_dai_fbcca": kF, "tranG_quyet_dinh": (nbc * u_trca + kF * mac_fb1) if kF >= 1 else None, "mot_dai_trca_gop_lop": u_trca, "so_dai_trong_tran": nbc, "tran_quyet_dinh": u_trca * nbc, "fbcca_mot_dai": mac_fb1,
               "tran1R_so_hai": hR, "tran1R_quyet_dinh": (u_trca + B.macs("fbcca", C, ns, h2=2 * hR)) if co_1R else None, "day_du_quyet_dinh": NB * u_trca + NB * mac_fb1, "gioi_thieu_moi_luot_khong_gioi_han": (nsub - 1) * NB * u_trca + NB * mac_fb1}
        res[str(T)] = {"tung_nguoi": per, "tb": tb, "mac": mac}
        if a.im_lang:
            print(T, "chay thong", sorted(tb.keys()), json.dumps(mac), flush=True)
        else:
            print(T, json.dumps({k: round(v, 4) for k, v in tb.items()}), json.dumps(mac), f"{time.time()-t0:.0f} s", flush=True)
    me = open(os.path.abspath(__file__), "rb").read()
    js = dict(seed=SEED, ds=a.ds, n_subjects=nsub, nguoi=list(range(1, nsub + 1)), kenh=(B.CH9 if a.ds != "dong" else ["POZ", "PO3", "PO4", "PO7", "PO8", "OZ", "O1", "O2"]), fs=fs, do_tre_s=c["lat"],
              dem_mac="phep nhan chinh moi quyet dinh: chieu + N tuong quan + chuan; khong tinh trong so dai, can, bank loc, hoc lai, LST, gioi thieu", n0=N0, k_rep=K_REP, m_on=M_ON, n_rec=N_REC, lo=LO, windows=WINS, cap=CAP,
              script_sha256=hashlib.sha256(me).hexdigest(), b1_diem_sha256=hashlib.sha256(open(B.__file__, "rb").read()).hexdigest(), ket_qua=res,
              ghi_chu="Tai hien RCS (Shi 2023) tren giao thuc phat lai theo thu tu tep; cua so co dinh; nhan yeu = quyet dinh da phat; "
                      "tran_* = chi TRCA gop lop, so dai vua tran 1 M; phan gioi thieu khong gioi han MAC. Khong phai so cua bai goc.")
    if not a.im_lang:
        json.dump(js, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)      # ghi nguyen tu
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
