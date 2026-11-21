#!/usr/bin/env python
"""THI NGHIEM B tren PS cua ZCU104. In MOT dong JSON moi lan goi.
Can: numpy, scipy, torch; cung thu muc: doi_thu_oacca.py, doi_thu_sfda.py, global_dnn.py, b1_diem.py, zcu104_b1gop_eg.py; du lieu tu
zcu104_doi_thu_goi.py (benchmark_S{1,17,27}.npz); trong so that fold017, fold027 (Benchmark 1,0 s, thu muc E4 results/b1v7_lo2/T10).
Luong: dat OMP/OPENBLAS/MKL_NUM_THREADS ngoai tien trinh (tep sweep) va torch.set_num_threads(--luong); in np.show_config va bien moi truong.
Phan:
  b1 : duong E4 day du tu EEG tho moi luot (trong so ngau nhien nhu cong R): dong ho bat dau TRUOC GD.tien_xu_ly (gom thiet ke bo loc trong loi goi)
       -> quyet dinh -> nen (lan truyen nguoc toi l2, EG, gop). 20 luot khoi dong may + --n luot, lap lai 240 luot tho cua nguoi 1.
  b2 : OACCA-NQ (doi_thu_oacca) tren nguoi 1, thu tu P1 r = 1: lam nong may 20 luot tren doi tuong rieng (bo), roi --phien phien x 240 luot,
       tao doi tuong moi dau moi phien. Dong ho bat dau TRUOC tien xu ly; quyet dinh va cap nhat tach bang hai ham sao chep nguyen van OACCA.buoc
       (kiem tuong duong voi OACCA.buoc tren mot phien truoc khi do: nhan va trang thai trung tung bit). Ghi nhanh (res1 = res6 hay khong) tung luot.
       Bao kem: tien xu ly voi bo loc tinh mot lan (cache), kiem trung ket qua.
  b3vong : mot vong 50 buoc SGDM (doi_thu_sfda.hoc_vong) tren workload TONG HOP (mang ngau nhien, X ngau nhien, nhan ngau nhien, lang gieng tinh
       bang lang_gieng tren workload do), a = --a, --lap lan.
  b3phu  : tau_phu tren du lieu va trong so THAT (fold --fold, a = --a): du_doan, silhouette, lang_gieng, FBCCA (a luot), sao chep trang thai.
  b3moc  : mot moc tron ven doi_thu_sfda.sfda_chon tren du lieu, trong so THAT (fold --fold, a = --a), seed torch.manual_seed(20261107 + 1000 fold + a),
       tran --tran-s giay (mac dinh 1.800): cham tran ghi "> 30 phut, chua hoan tat". Ghi so vong, nhanh, lam; bo nho: byte dau vao, ban sao trang thai,
       kich hoat mot buoc hoc (storage duy nhat qua saved_tensors_hooks), ma tran a x a; VmHWM bao kem.
"""
import argparse, copy, io, json, os, signal, sys, time, contextlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FS, ON = 250, 125


def q(v):
    v = np.asarray(v, float)
    return dict(trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)), max_ms=1e3 * float(v.max()),
                tb_ms=1e3 * float(v.mean()), n=int(len(v)), vuot_50ms=float((v > 0.05).mean()), vuot_500ms=float((v > 0.5).mean()))


def vmhwm_kb():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"):
            return int(l.split()[1])
    return -1


def moi_truong():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    return dict(numpy=np.__version__, torch=torch.__version__, torch_luong=torch.get_num_threads(),
                env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                np_config=buf.getvalue()[-1500:])


def nap(du_lieu, s):
    z = np.load(os.path.join(du_lieu, f"benchmark_S{s}.npz"))
    return {k: z[k] for k in z.files}


# ---------------- B1
def phan_b1(a):
    import b1_diem as B, global_dnn as GD
    import zcu104_b1gop_eg as Z
    import math
    z = nap(a.du_lieu, 1); Xr = z["X"]
    ns = int(round(a.T * FS)); C, K = 9, 5
    torch.manual_seed(a.seed)
    m = Z.GlobalDNN(C, ns, 40).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W0, b0 = m.l2.weight.clone(), m.l2.bias.clone()
    DW = 0.01 * torch.randn(K, W0.numel()); DB = 0.01 * torch.randn(K, b0.numel())
    theta = torch.zeros(K); Gm = 0.0
    W, b = W0 + DW.mean(0).view_as(W0), b0 + DB.mean(0)
    tb, tq, tn, tt = [], [], [], []
    for i in range(20 + a.n):
        xr = Xr[i % 240][None]
        t0 = time.perf_counter()
        x = torch.tensor(GD.tien_xu_ly(xr, "benchmark", "causal", ns, FS))[0][None]   # (1, 3, C, ns)
        t1 = time.perf_counter()
        Wl, bl = W.detach().requires_grad_(True), b.detach().requires_grad_(True)
        with torch.no_grad():
            h1 = m.l1(x)
        lg = m.sau(h1, Wl, bl); int(lg.argmax(1))
        t2 = time.perf_counter()
        gW, gb = torch.autograd.grad(-Z.do_tin(lg).sum(), [Wl, bl])
        g = DW @ gW.flatten() + DB @ gb
        Gm = max(Gm, float(g.abs().max()))
        if Gm > 0:
            theta = theta - math.sqrt(2.0 * math.log(K) / (Gm * Gm * 200)) * g
        lam = torch.softmax(theta, 0)
        W = W0 + (lam @ DW).view_as(W0); b = b0 + lam @ DB
        t3 = time.perf_counter()
        if i >= 20:
            tb.append(t1 - t0); tq.append(t2 - t0); tn.append(t3 - t2); tt.append(t3 - t0)
    return dict(bank_loc=q(tb), quyet_dinh_tu_tho=q(tq), nen=q(tn), tong_cung_luot=q(tt))


# ---------------- B2
def tach_oacca(O):
    """Hai ham sao chep NGUYEN VAN OACCA.buoc, tach quyet dinh va cap nhat."""
    from scipy.linalg import eigh
    NB, FB = O.NB, O.FB

    def quyet_dinh(m, sig):
        rho1 = np.empty((NB, m.F)); rho5 = np.empty_like(rho1); rho6 = np.empty_like(rho1); Xs, As = [], []
        for k in range(NB):
            X = sig[k] - sig[k].mean(1, keepdims=True)
            sd_ = X.std(1, ddof=1, keepdims=True)
            assert (sd_ > 0).all(), "kenh hang so"
            X = X / sd_
            r1, A = m.diem(X)
            r2 = np.zeros(m.F); r3 = np.zeros(m.F)
            if m.co_om:
                a_ = m.u[k] @ X; a_ = a_ - a_.mean()
                b_ = np.einsum("fhn,h->fn", m.Yc, m.v[k])
                r2 = (b_ @ a_) / (np.linalg.norm(a_) * np.linalg.norm(b_, axis=1))
            if m.co_psf:
                p = m.w[k] @ X; p = p - p.mean()
                r3 = np.linalg.norm(np.einsum("fnh,n->fh", m.Qy, p), axis=1) / np.linalg.norm(p)
            rho1[k], rho5[k], rho6[k] = r1, r1 + r3, r1 + r2 + r3
            Xs.append(X); As.append(A)
        res1, res5, res6 = (int(np.argmax(FB @ r)) for r in (rho1, rho5, rho6))
        return (res1, res5, res6), Xs, As

    def cap_nhat(m, res, Xs, As):
        res1, res5, res6 = res
        for k in range(NB):
            if res1 == res6:
                s = As[k][:, res1]; s = s / np.linalg.norm(s)
                m.Cpsf[k] += np.outer(s, s)
                m.w[k] = eigh(m.Cpsf[k])[1][:, -1]
            m.Cxx[k] += Xs[k] @ Xs[k].T
            m.Cxy[k] += Xs[k] @ m.Y[res5].T
            C, H2 = m.C, m.H2
            Am = np.zeros((C + H2, C + H2)); Am[:C, C:] = m.Cxy[k]; Am[C:, :C] = m.Cxy[k].T
            Bm = np.zeros_like(Am); Bm[:C, :C] = m.Cxx[k]; Bm[C:, C:] = np.eye(H2)
            ve = eigh(Am, Bm)[1][:, -1]
            u1, v1 = ve[:C].copy(), ve[C:]
            if u1[0] == 1:
                u1 = np.zeros(C); u1[-3:] = 1
            m.u[k], m.v[k] = u1, v1
        if res1 == res6:
            m.co_psf = True
        m.co_om = True
    return quyet_dinh, cap_nhat


def phan_b2(a):
    import b1_diem as B
    import doi_thu_oacca as O
    z = nap(a.du_lieu, 1); Xr, pi = z["X"], z["pi_P1_r1"]
    ns = int(round(a.T * FS)); lat = int(round(B.DS["benchmark"]["lat"] * FS)); nmax = int(round(1.0 * FS))
    Y, Qy = O.tham_chieu(z["freqs"], z["phases"], ns, FS)
    tx = lambda xr: O.tien_xu_ly(xr, "nq", ON, lat, nmax, FS)[0][:, :, :ns]          # nhu doi_thu_oacca nq: loc toi 1,0 s roi cat
    qd, cn = tach_oacca(O)
    # kiem tuong duong voi OACCA.buoc tren mot phien (khong tinh gio)
    m1, m2 = O.OACCA(9, Y, Qy), O.OACCA(9, Y, Qy)
    for t in pi:
        s_ = tx(Xr[t][None])
        r_a = m1.buoc(s_); r_b, Xs, As = qd(m2, s_); cn(m2, r_b, Xs, As)
        assert r_a == r_b
    for k_ in ("Cxx", "Cxy", "Cpsf", "u", "v", "w"):
        assert np.array_equal(getattr(m1, k_), getattr(m2, k_)), k_
    # bo loc cache (bao kem): kiem trung ket qua
    (bn, an), sos = O.bo_loc(FS)
    from scipy.signal import lfilter, sosfilt
    def tx_cache(xr):
        st = ON + lat
        Xb = xr[:, :, : st + nmax] - xr[:, :, :ON].mean(axis=2, keepdims=True)
        Xb = lfilter(bn, an, Xb, axis=2)
        return np.stack([sosfilt(so, Xb, axis=2)[:, :, st: st + nmax] for so in sos], 1)[0][:, :, :ns]
    assert np.array_equal(tx_cache(Xr[pi[0]][None]), tx(Xr[pi[0]][None]))
    mw = O.OACCA(9, Y, Qy)                                                        # lam nong may: doi tuong rieng, bo
    for t in pi[:20]:
        r_, Xs, As = qd(mw, tx(Xr[t][None])); cn(mw, r_, Xs, As)
    tq, tqc, tc, tt, tcache, nhanh = [], [], [], [], [], []
    for _ in range(a.phien):
        m = O.OACCA(9, Y, Qy)                                                     # reset dau moi phien
        for t in pi:
            xr = Xr[t][None]
            t0 = time.perf_counter()
            s_ = tx(xr)
            t1 = time.perf_counter()
            r_, Xs, As = qd(m, s_)
            t2 = time.perf_counter()
            cn(m, r_, Xs, As)
            t3 = time.perf_counter()
            t4 = time.perf_counter(); tx_cache(xr); t5 = time.perf_counter()
            tq.append(t2 - t0); tqc.append(t2 - t1); tc.append(t3 - t2); tt.append(t3 - t0); tcache.append(t5 - t4); nhanh.append(r_[0] == r_[2])
    nhanh = np.array(nhanh); tc = np.array(tc); tt = np.array(tt)
    theo_nhanh = {}
    for ten, mk in (("res1_bang_res6", nhanh), ("res1_khac_res6", ~nhanh)):
        theo_nhanh[ten] = dict(so_mau=int(mk.sum()), cap_nhat=(q(tc[mk]) if mk.any() else "khong co mau"), tong_cung_luot=(q(tt[mk]) if mk.any() else "khong co mau"))
    return dict(quyet_dinh_tu_tho=q(tq), quyet_dinh_loi=q(tqc), cap_nhat=q(tc), tong_cung_luot=q(tt), tien_xu_ly_cache=q(tcache),
                theo_nhanh=theo_nhanh, tuong_duong_buoc="dat")


# ---------------- B3
class DemLuu:
    def __init__(self, loai):
        self.loai = set(loai); self.ptr, self.byte = set(), 0

    def goi(self, t):
        st = t.untyped_storage(); p = st.data_ptr()
        if p not in self.loai and p not in self.ptr:
            self.ptr.add(p); self.byte += st.nbytes()
        return t


def mang_va_du_lieu(a, that):
    import global_dnn as GD, b1_diem as B
    import doi_thu_sfda as S
    ns = 250
    if that:
        import hashlib
        model = GD.GlobalDNN(9, ns, nch=S.NCH, p_cuoi=0.95)
        f = os.path.join(a.trongso, f"fold{a.fold:03d}_trongso.pt")
        mf = json.load(open(os.path.join(a.du_lieu, "manifest.json")))         # doi chieu trong so voi JSON d6
        assert hashlib.sha256(open(f, "rb").read()).hexdigest() == mf["trongso_sha256"][str(a.fold)], f"trong so fold {a.fold} khac manifest d6"
        model.load_state_dict(torch.load(f, map_location="cpu"))
        z = nap(a.du_lieu, a.fold)
        X = torch.tensor(GD.tien_xu_ly(z["X"][:a.a], "benchmark", "causal", ns, FS))   # cat truoc khi tao tensor
        return model, X, z, S, B, GD
    torch.manual_seed(a.seed)                                                   # seed TRUOC khi khoi tao mang tong hop
    model = GD.GlobalDNN(9, ns, nch=S.NCH, p_cuoi=0.95)
    return model, torch.randn(a.a, 3, 9, ns), None, S, B, GD


def phan_b3vong(a):
    model, X, _, S, _, _ = mang_va_du_lieu(a, False)
    for p in model.parameters():
        p.requires_grad_(False)
    sd0 = copy.deepcopy(model.state_dict())
    T = torch.randint(0, 40, (a.a,))
    _, ind, cap = S.lang_gieng(model, X, T, 0.6)
    t = []
    for _ in range(a.lap):
        t0 = time.perf_counter(); S.hoc_vong(model, sd0, X, T, ind, cap, 0.6, a.a, 0.6); t.append(time.perf_counter() - t0)
    return dict(a=a.a, loai="workload tong hop", seed=a.seed, mot_vong_s=[float(v) for v in t], trung_vi_s=float(np.median(t)))


def phan_b3phu(a):
    model, X, z, S, B, GD = mang_va_du_lieu(a, True)
    for p in model.parameters():
        p.requires_grad_(False)
    out = {}
    for _ in range(a.lap):
        t0 = time.perf_counter(); Y = S.du_doan(model, X); t1 = time.perf_counter()
        S.silhouette(model, X, Y); t2 = time.perf_counter()
        S.lang_gieng(model, X, Y, 0.6); t3 = time.perf_counter()
        S.fbcca(z["X"][:a.a], "benchmark", z["freqs"], 250, B); t4 = time.perf_counter()
        copy.deepcopy(model.state_dict()); t5 = time.perf_counter()
        for k, v in (("du_doan", t1 - t0), ("silhouette", t2 - t1), ("lang_gieng", t3 - t2), ("fbcca", t4 - t3), ("sao_chep_trang_thai", t5 - t4)):
            out.setdefault(k, []).append(float(v))
    return dict(a=a.a, fold=a.fold, loai="du lieu va trong so that", **{k: dict(lan_s=v, trung_vi_s=float(np.median(v))) for k, v in out.items()})


class HetGio(Exception):
    pass


def phan_b3moc(a):
    """Mot moc tron ven: dong ho va tran bao FBCCA (tau_phu) + sfda_chon. Bo nho do o lan chay rieng b3bonho."""
    model, X, z, S, B, GD = mang_va_du_lieu(a, True)
    for p in model.parameters():
        p.requires_grad_(False)
    sd0 = copy.deepcopy(model.state_dict())
    Xr = z["X"][:a.a]
    torch.manual_seed(20261107 + 1000 * a.fold + a.a)
    def het(*_):
        raise HetGio()
    signal.signal(signal.SIGALRM, het); signal.alarm(a.tran_s)
    t0 = time.perf_counter()
    try:
        fb = torch.tensor(S.fbcca(Xr, "benchmark", z["freqs"], 250, B), dtype=torch.long)
        t_fb = time.perf_counter() - t0
        kq, tt = S.sfda_chon(model, sd0, X, fb, S.P2["benchmark"], 40)
        signal.alarm(0)
        moc = dict(trang_thai="hoan tat", tg_s=time.perf_counter() - t0, fbcca_s=t_fb, so_lan=tt["so_lan"], tong_vong=int(sum(tt["so_lan"])),
                   lam=tt["lam"], nhanh=tt["nhanh"], co_mang=tt["co_mang"])
    except HetGio:
        moc = dict(trang_thai="> %d s, chua hoan tat" % a.tran_s, tg_s=time.perf_counter() - t0,
                   ghi_chu="SIGALRM chi xu ly giua hai phep toan C++; tg_s la thoi diem ngat thuc te")
    return dict(a=a.a, fold=a.fold, seed=20261107 + 1000 * a.fold + a.a, loai="du lieu va trong so that", moc=moc, vmhwm_kb=vmhwm_kb())


def phan_b3bonho(a):
    """Lan chay RIENG (khong tinh gio), cung seed voi b3moc. Ba so, ghi ro loai:
    (1) dau_vao_X_byte: storage dau vao a luot SFDA phai giu; so CHINH XAC va la CAN DUOI cua bo nho them (dung de so voi R3 = 1 MB).
    (2) uoc_luong_diem_chup: tap storage cung song tai cac diem chup (cuoi luot xuoi va sau lan truyen nguoc o buoc thu hai moi vong;
        sau moi loi goi silhouette, lang_gieng), gom tensor Python (gc), gradient, dem momentum, kich hoat dang duoc do thi giu (dem song
        bang pack hook, tu tru khi do thi giai phong); cong them tam a x a trong lang_gieng dem theo ma. La UOC LUONG tai diem chup,
        khong phai dinh bo cap phat.
    (3) rss_tang_them_dinh_byte: dinh RSS (lay mau /proc/self/statm moi 10 ms) tru RSS ngay truoc sfda_chon; gom bo nho dem cua bo cap phat;
        bao kem."""
    import gc, threading
    model, X, z, S, B, GD = mang_va_du_lieu(a, True)
    for p in model.parameters():
        p.requires_grad_(False)
    sd0 = copy.deepcopy(model.state_dict())
    fb = torch.tensor(S.fbcca(z["X"][:a.a], "benchmark", z["freqs"], 250, B), dtype=torch.long)
    kh_song = {}                                                               # storage ptr -> [so lan dang giu, byte]

    class Giu:
        def __init__(self, t_):
            self.t = t_; st = t_.untyped_storage(); self.p = st.data_ptr()
            e = kh_song.setdefault(self.p, [0, st.nbytes()]); e[0] += 1

        def __del__(self):
            e = kh_song.get(self.p)
            if e is not None:
                e[0] -= 1
                if e[0] <= 0:
                    del kh_song[self.p]

    def song(them=()):
        d = {}
        for o in gc.get_objects():
            if isinstance(o, torch.Tensor):
                try:
                    st = o.untyped_storage(); d[st.data_ptr()] = st.nbytes()
                except Exception:
                    pass
        for t_ in them:
            if isinstance(t_, torch.Tensor):
                d[t_.untyped_storage().data_ptr()] = t_.untyped_storage().nbytes()
        for p_, (n_, b_) in kh_song.items():
            d[p_] = b_
        return d
    nen = song(); dinh = dict(byte=0, tang_them=0, diem=None)

    def chup(ten, them=(), tam=0):
        d = song(them)
        tong = sum(d.values()) + tam; tt = sum(v for k, v in d.items() if k not in nen) + tam
        if tt > dinh["tang_them"]:
            dinh.update(byte=tong, tang_them=tt, diem=ten)

    def hoc_vong_do(model_, sd, X_, T, ind, cap, lam, mau_so, p2):
        model_.load_state_dict(sd)
        for p in model_.parameters():
            p.requires_grad_(True)
        model_.d1.p = p2; model_.d2.p = p2
        ps = list(model_.named_parameters())
        opt = torch.optim.SGD([{"params": [p for n, p in ps if n.endswith("weight")], "weight_decay": 1e-3},
                               {"params": [p for n, p in ps if n.endswith("bias")], "weight_decay": 0.0}], lr=1e-4, momentum=0.9)
        model_.train()
        for j_ in range(S.J_BUOC):
            opt.zero_grad(set_to_none=True)
            if j_ == 1:
                dem = [v for s_ in opt.state.values() for v in s_.values() if isinstance(v, torch.Tensor)]
                with torch.autograd.graph.saved_tensors_hooks(lambda t_: Giu(t_), lambda g_: g_.t):
                    loss = S.mat_mat(model_(X_), T, ind, cap, lam, mau_so)[0]
                chup("cuoi luot xuoi", dem)
                loss.backward()
                chup("sau lan truyen nguoc", dem + [p.grad for p in model_.parameters() if p.grad is not None])
                del loss
                opt.step()
            else:
                S.mat_mat(model_(X_), T, ind, cap, lam, mau_so)[0].backward(); opt.step()
        model_.eval()
        for p in model_.parameters():
            p.requires_grad_(False)
        return copy.deepcopy(model_.state_dict())
    lg_goc, sil_goc = S.lang_gieng, S.silhouette
    L_ = a.a; tam_lg = L_ * L_ * 4 * 2 + L_ * L_ * 8 + L_ * (L_ - 2) * (4 + 8)    # R, sc (float32), si (int64), diffs (float32), nho (int64)

    def lang_gieng_do(*ar, **kw):
        r_ = lg_goc(*ar, **kw); chup("sau lang_gieng (cong tam a x a dem theo ma)", tam=tam_lg); return r_

    def silhouette_do(*ar, **kw):
        r_ = sil_goc(*ar, **kw); chup("sau silhouette"); return r_
    S.hoc_vong, S.lang_gieng, S.silhouette = hoc_vong_do, lang_gieng_do, silhouette_do
    trang = os.sysconf("SC_PAGE_SIZE")
    rss = lambda: int(open("/proc/self/statm").read().split()[1]) * trang
    rss0 = rss(); dinh_rss = [rss0]; dung = threading.Event()

    def lay_mau():
        while not dung.is_set():
            dinh_rss[0] = max(dinh_rss[0], rss()); time.sleep(0.01)
    th = threading.Thread(target=lay_mau, daemon=True); th.start()
    torch.manual_seed(20261107 + 1000 * a.fold + a.a)

    def het(*_):
        raise HetGio()
    signal.signal(signal.SIGALRM, het); signal.alarm(a.tran_s)
    try:
        S.sfda_chon(model, sd0, X, fb, S.P2["benchmark"], 40); signal.alarm(0); tt = "hoan tat"
    except HetGio:
        tt = "> %d s, chua hoan tat (so do tren cac vong da chay)" % a.tran_s
    dung.set(); th.join()
    return dict(a=a.a, fold=a.fold, loai="du lieu va trong so that", trang_thai=tt,
                dau_vao_X_byte=int(X.untyped_storage().nbytes()),
                uoc_luong_diem_chup=dict(tong_byte=int(dinh["byte"]), tang_them_byte=int(dinh["tang_them"]), diem=dinh["diem"], tam_lang_gieng_dem_theo_ma=tam_lg),
                rss_tang_them_dinh_byte=int(dinh_rss[0] - rss0), vmhwm_kb=vmhwm_kb())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phan", required=True, choices=["b1", "b2", "b3vong", "b3phu", "b3moc", "b3bonho"])
    ap.add_argument("--du-lieu", default="du_lieu_ben"); ap.add_argument("--trongso", default="trongso_T10")
    ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--luong", type=int, default=1)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--phien", type=int, default=5)
    ap.add_argument("--a", type=int, default=40); ap.add_argument("--fold", type=int, default=17)
    ap.add_argument("--lap", type=int, default=5); ap.add_argument("--tran-s", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=20261107); ap.add_argument("--tien-trinh", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.luong)
    nhiet = lambda: open("/sys/class/thermal/thermal_zone0/temp").read().strip() if os.path.exists("/sys/class/thermal/thermal_zone0/temp") else None
    tan_so = lambda: open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read().strip() if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") else None
    dau = dict(nhiet=nhiet(), tan_so=tan_so())
    kq = dict(b1=phan_b1, b2=phan_b2, b3vong=phan_b3vong, b3phu=phan_b3phu, b3moc=phan_b3moc, b3bonho=phan_b3bonho)[a.phan](a)
    print(json.dumps(dict(phan=a.phan, T=a.T, luong=a.luong, tien_trinh=a.tien_trinh, seed=a.seed,
                          dau=dau, cuoi=dict(nhiet=nhiet(), tan_so=tan_so()), moi_truong=moi_truong(), ket_qua=kq, vmhwm_kb=vmhwm_kb())), flush=True)


if __name__ == "__main__":
    main()
