#!/usr/bin/env python
"""NHANH SONG SONG "GOP HE SO": thich nghi khong nhan trong khong gian HE SO GOP mo-dun nguoi nguon.
Benchmark la bo PHAT TRIEN.

Dung lai nguyen ham cua b1moi_lai.py (job 18691): bo trich = Global DNN thu nho (nch 22, loc nhan qua) cua lo 2, CO DINH; mo-dun kieu B = so gia lop
loc khong gian l2 (220 tham so) cua tung nguoi nguon; 40 luot dau quyet dinh bang mang chung W0; chon nguon bang do tin khong nhan cua 40 luot dau.
W(lam) = W0 + sum_i lam_i * Delta_i, lam tren don hinh cua K nguon duoc chon. Suy luan luon la MOT mang (he so gop vao tham so l2).
Phat lai nhan qua theo thu tu khoi roi muc tieu trong tep (KHONG phai thu tu thu that); moi luot duoc giai bang tham so hien co roi moi dung de cap nhat.
Cac o:
  E0  : SF_B = lam deu tren 5 nguon, giu co dinh (hoi quy job 18691).
  E1  : SU_B cua job 18691: 220 tham so tu do, nhan gia CU (gom 40 luot dau do W0 phat), Adam theo moc, moc dau o luot 40 (hoi quy job 18691).
  E1b : nhu E1 nhung chi nhan gia MOI (luot tu 41, do mo-dun ca nhan phat; cung nguong do tin), moc dau co cap nhat o luot 60.
  E2  : nhu E1 (nhan cu) nhung tham so la theta (K = 5), lam = softmax(theta), lr LR_THETA_MOC (0,01).
  E2b : nhu E1b (nhan moi) nhung tham so la theta.
  E3  : nhu E2b (cung quy tac chon luot; tap luot thuc te co the khac tu moc 80), mat mat = -do_tin (khong dung nhan gia).
  E1d : dung lai LUONG nhan va do tin cua E1 (cung chi so luot o moi moc, cung nhan sau luot 40); chi NHAN HOC cua 40 luot dau do SF_B phat.
  E1c, E2c : 220 tham so tu do / theta, CUNG nhan va CUNG tap luot tu bo phat co dinh E0 (luot tu 41; nhan qua) - tach rieng khong gian.
  all_seq : doi chung toan mang nhu job 18691 (ban bat dropout, cung seed).
  E4 : Gradient mu truc tuyen tung luot trong mien log: theta <- theta - eta_t g_t, lam = softmax(theta), theta0 = 0,
        g_t = d(-do_tin(z_t(lam_t)))/d lam, eta_t = sqrt(2 ln K / (G_t^2 T)), G_t = cuc dai |g| da gap sau luot 40 (ke ca luot t), T = so luot sau N0.
  E5  : E4 voi K = 11 (bao kem); E5f = deu tren 11 nguon, giu co dinh (tham chieu P4 cua E5).
  E6  : E4 voi 5 nguon chon bang CHAM BAC NHAT (TB_t <grad_W c_t(W0), Delta_i>, mot luot lan truyen nguoc cho moi nguon); E6f = cung nguon, giu co dinh.
  O5, O11 : chan doan: theta khop bang NHAN THAT cua 40 luot dau (100 buoc Adam, lr LR_THETA = 0,1), giu co dinh. Tinh SAU CUNG, khong anh huong o nao.
Nhan that chi dung de cham, cho O5/O11, de bao chat luong tung mo-dun va do dung nhan gia (P3).
"""
import argparse, json, math, os, sys, time, hashlib
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402
import b1moi_lai as L        # noqa: E402
import du_lieu_lee as DL      # noqa: E402  (them 2026-11-15, buoc 2b)

for _ds in B.DS:                      # may khac (maychu4) khong doc duoc thu muc cua gv_dnton: B1_ROOT_BENCHMARK, B1_ROOT_BETA tro toi ban sao da doi chieu md5
    B.DS[_ds]["root"] = os.environ.get("B1_ROOT_" + _ds.upper(), B.DS[_ds]["root"])

SEED = L.SEED
N0, LO, NCH = L.N0, L.LO, L.NCH
K_CHINH, K_RONG = 5, 11
LR_THETA_MOC = 0.01  # o theo moc tren theta (E2, E2b, E2c, E3): 10 buoc Adam moi moc dich theta toi da khoang 0,1, cung co buoc EG dau (0,127)
LR_THETA, BUOC_ORACLE = 0.1, 100     # chi cho chan doan O5, O11
HE_SO_ETA = (1.0, 0.5, 2.0)          # 1.0 = ung vien; 0.5 va 2.0 chi bao cao, khong chon


def gop(DW, DB, lam):
    return {"dW": (lam[:, None, None, None, None] * DW).sum(0), "db": (lam[:, None] * DB).sum(0)}


def xep(mods, idx):
    return torch.stack([mods[int(j)]["dW"] for j in idx]), torch.stack([mods[int(j)]["db"] for j in idx])


def tv(lam):
    return float(0.5 * np.abs(np.asarray(lam) - 1.0 / len(lam)).sum())


def kc(m1, m2):
    return float(torch.sqrt(((m1["dW"] - m2["dW"]) ** 2).sum() + ((m1["db"] - m2["db"]) ** 2).sum()))


def phat_lai_co_dinh(model, h1, lg_c, mod):
    n = h1.shape[0]; dec = torch.empty(n, dtype=torch.long, device=h1.device)
    dec[:N0] = lg_c[:N0].argmax(1)
    with torch.no_grad():
        dec[N0:] = L.sau(model, h1[N0:], mod).argmax(1)
    return dec.cpu().numpy()


def phat_lai_moc(model, h1, lg_c, yt, ms, DW, DB, khong_gian, nhan, mat_mat, lab_dau=None, ngoai=None):
    """Cap nhat theo moc 20 luot, BUOC_CN buoc Adam tao lai moi moc, tham so giu am qua cac moc (nhu L.phat_lai_mod).
    khong_gian: 'W' (220 tham so tu do, bat dau tu ms, lr L.LR_MOD) | 'theta' (lam = softmax(theta), theta0 = 0, lr LR_THETA_MOC).
    nhan: 'cu' (moi luot da qua, gom 40 luot dau) | 'moi' (chi luot tu N0). Nguong do tin = phan vi 25 cua do tin W0 tren 40 luot dau.
    mat_mat: 'nhangia' (entropy cheo voi nhan hoc) | 'dotin' (-do_tin, khong dung nhan gia).
    lab_dau (N0,): thay NHAN HOC cua 40 luot dau (chi trong mat mat; mat na va lich giu nguyen) - o E1d.
    ngoai = (lab, cf): nhan hoc va do tin lay tu MOT luong co dinh, nhan qua: E1c, E2c dung luong E0; E1d dung luong cua E1 (kem lab_dau).
    Tra (dec, cf, them); them['van_tay'] = sha1 cua chi so luot va nhan hoc o tung moc (de kiem "cung du lieu hoc")."""
    n = h1.shape[0]; dev = h1.device
    dec = torch.empty(n, dtype=torch.long, device=dev); dec[:N0] = lg_c[:N0].argmax(1)
    cf = torch.empty(n, device=dev); cf[:N0] = L.do_tin(lg_c[:N0])
    nguong = torch.quantile(cf[:N0], 0.25)
    mod = {k: v.clone() for k, v in ms.items()}; theta = torch.zeros(DW.shape[0], device=dev)
    q_nhan, q_dau, so_luot, van_tay, so_lan = [], [], [], [], 0
    for a in range(N0, n, LO):
        dau = 0 if nhan == "cu" else N0
        if ngoai is None:
            lab_src = dec.clone(); cf_src = cf
        else:
            lab_src = ngoai[0].clone(); cf_src = ngoai[1]
        if lab_dau is not None:
            lab_src[:N0] = lab_dau
        idx = dau + torch.where(cf_src[dau:a] >= nguong)[0]
        if len(idx) > 0:
            lab = lab_src[idx]
            so_lan += 1; so_luot.append(int(len(idx))); q_nhan.append(float((lab == yt[idx]).float().mean()))      # P3: chi de bao cao
            k40 = idx < N0
            q_dau.append(float((lab[k40] == yt[idx[k40]]).float().mean()) if bool(k40.any()) else None)            # P3 rieng cho nhan khoi dong
            van_tay.append(hashlib.sha1(idx.cpu().numpy().tobytes() + lab.cpu().numpy().tobytes()).hexdigest())
            if khong_gian == "W":
                assert mat_mat == "nhangia"
                mod = L.hoc(model, h1[idx], lab, mod, L.BUOC_CN, L.LR_MOD)
            else:
                th = theta.clone().requires_grad_(True)
                opt = torch.optim.Adam([th], lr=LR_THETA_MOC)
                for _ in range(L.BUOC_CN):
                    opt.zero_grad(set_to_none=True)
                    lg = L.sau(model, h1[idx], gop(DW, DB, torch.softmax(th, 0)))
                    mm = Fn.cross_entropy(lg, lab) if mat_mat == "nhangia" else -L.do_tin(lg).mean()
                    mm.backward(); opt.step()
                theta = th.detach(); mod = gop(DW, DB, torch.softmax(theta, 0))
        with torch.no_grad():
            lg = L.sau(model, h1[a:a + LO], mod)
        dec[a:a + LO] = lg.argmax(1); cf[a:a + LO] = L.do_tin(lg)
    lam = torch.softmax(theta, 0).cpu().numpy()
    them = dict(so_lan_cap_nhat=so_lan, so_luot_hoc=so_luot, van_tay=van_tay, q_nhan_gia=q_nhan, q_nhan_khoi_dong=q_dau, dich_W=kc(mod, ms),
                chuan_ms=kc(ms, {"dW": torch.zeros_like(ms["dW"]), "db": torch.zeros_like(ms["db"])}))
    if khong_gian == "theta":
        them.update(lam=lam.tolist(), tv=tv(lam))
    return dec.cpu().numpy(), cf, them


def phat_lai_eg(model, h1, lg_c, DW, DB, he_so):
    """E4, E5, E6: gradient mu truc tuyen tung luot (mien log), mat mat -do_tin cua CHINH luot vua giai."""
    K = DW.shape[0]; n = h1.shape[0]; dev = h1.device; T = n - N0
    dec = torch.empty(n, dtype=torch.long, device=dev); dec[:N0] = lg_c[:N0].argmax(1)
    theta = torch.zeros(K, device=dev); G = 0.0; eta = 0.0; bo_khong_huu_han = 0; bo_G0 = 0
    for t in range(N0, n):
        lv = torch.softmax(theta, 0).requires_grad_(True)
        lg = L.sau(model, h1[t:t + 1], gop(DW, DB, lv))
        dec[t] = lg.argmax(1)[0]                                   # quyet dinh bang lam_t TRUOC khi cap nhat
        (g,) = torch.autograd.grad(-L.do_tin(lg).sum(), lv)
        if not torch.isfinite(g).all():  # bo cap nhat cua luot nay, dem lai
            bo_khong_huu_han += 1; continue
        G = max(G, float(g.abs().max()))
        if G > 0:
            eta = he_so * math.sqrt(2.0 * math.log(K) / (G * G * T))
            theta = theta - eta * g
        else:
            bo_G0 += 1
    lam = torch.softmax(theta, 0).cpu().numpy()
    return dec.cpu().numpy(), dict(lam=lam.tolist(), tv=tv(lam), G=G, eta_cuoi=eta, dich_W=kc(gop(DW, DB, torch.softmax(theta, 0)), gop(DW, DB, torch.full((K,), 1.0 / K, device=dev))),
                                   bo_khong_huu_han=bo_khong_huu_han, bo_G0=bo_G0)


def khop_nhan_that(model, h1, yt, lg_c, DW, DB):
    th = torch.zeros(DW.shape[0], device=h1.device, requires_grad=True)
    opt = torch.optim.Adam([th], lr=LR_THETA)
    for _ in range(BUOC_ORACLE):
        opt.zero_grad(set_to_none=True)
        Fn.cross_entropy(L.sau(model, h1[:N0], gop(DW, DB, torch.softmax(th, 0))), yt[:N0]).backward(); opt.step()
    lam = torch.softmax(th.detach(), 0)
    return phat_lai_co_dinh(model, h1, lg_c, gop(DW, DB, lam)), dict(lam=lam.cpu().numpy().tolist(), tv=tv(lam.cpu().numpy()))


def cham_bac_nhat(model, h1, mods, C, dev):
    """Diem xap xi cua tung mo-dun nguon tren 40 luot dau: TB_t <grad_W c_t(W0), Delta_i> (hang so c_t(W0) chung cho moi nguon nen bo)."""
    DWa, DBa = xep(mods, range(len(mods)))
    diem = torch.zeros(len(mods), device=dev)
    for t in range(N0):
        m0 = {k: v.clone().requires_grad_(True) for k, v in L.mod_chung("B", C, dev).items()}
        c = L.do_tin(L.sau(model, h1[t:t + 1], m0)).sum()
        gW, gb = torch.autograd.grad(c, [m0["dW"], m0["db"]])
        diem += (DWa * gW[None]).flatten(1).sum(1) + (DBa * gb[None]).sum(1)
    return diem / N0


def chay_fold(i, D, a, dev, wdir):
    seed = SEED + 1000 * (i + 1) + int(round(a.T * 10))
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    nsub = len(D["X"]); src = [s for s in range(nsub) if s != i]
    C, ns = D["X"][0].shape[2], D["X"][0].shape[3]
    model = GD.GlobalDNN(C, ns, nch=NCH, p_cuoi=0.95).to(dev)
    model.load_state_dict(torch.load(os.path.join(wdir, f"fold{i+1:03d}_trongso.pt"), map_location=dev))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xu = torch.tensor(D["X"][i], device=dev); y = D["y"][i]; yt = torch.tensor(y, device=dev, dtype=torch.long)
    mc = L.mod_chung("B", C, dev)
    with torch.no_grad():
        h1u = L.truoc(model, Xu); lg0 = model(Xu)
        lg_c = L.sau(model, h1u, mc)                                    # nhu job 18691: 40 luot dau quyet dinh bang logit cua mo-dun chung
        assert torch.allclose(lg0, lg_c, atol=1e-4)
    mods = []
    for s in src:                                                       # mo-dun nguon kieu B, nhu job 18691 (L.hoc tat dinh, khong dung so ngau nhien)
        Xs = torch.tensor(D["X"][s], device=dev); ys = torch.tensor(D["y"][s], device=dev, dtype=torch.long)
        with torch.no_grad():
            h1s = L.truoc(model, Xs)
        mods.append(L.hoc(model, h1s, ys, mc, L.BUOC_NGUON, L.LR_MOD))
    with torch.no_grad():
        lgs = torch.stack([L.sau(model, h1u, m) for m in mods])        # (S, n, N)
    g = L.do_tin(lgs[:, :N0]).mean(1)                                   # khong dung nhan
    thu_tu = torch.argsort(-g, stable=True)
    top5, top11 = thu_tu[:K_CHINH], thu_tu[:K_RONG]
    qs = (lgs[:, N0:].argmax(2).cpu().numpy() == y[None, N0:]).mean(1)  # chat luong tung mo-dun (nhan CHI de bao cao)
    res = {"fold": i + 1, "seed": seed, "nguon_chon5": [src[int(j)] + 1 for j in top5], "nguon_chon11": [src[int(j)] + 1 for j in top11],
           "q_nguon_tb": float(qs.mean()), "q_chon5": [float(qs[int(j)]) for j in top5], "q_nguon_tot_nhat": float(qs.max()),
           "tinh": L.cham(lg_c.argmax(1).cpu().numpy(), y)}
    DW5, DB5 = xep(mods, top5); DW11, DB11 = xep(mods, top11)
    deu5 = torch.full((K_CHINH,), 1.0 / K_CHINH, device=dev)
    ms = {k: torch.stack([mods[int(j)][k] for j in top5]).mean(0) for k in mc}   # DUNG stack().mean(0) nhu job 18691: gop(.., 0,2) lech ~1e-8 va doi E1 1-4 quyet dinh (chan doan job 18738)
    assert all(torch.allclose(ms[k], gop(DW5, DB5, deu5)[k], atol=1e-6) for k in mc)
    def ghi(ten, dec, them=None):                                       # luu ca chuoi quyet dinh de tinh P4 (ti le quyet dinh doi so voi o tham chieu)
        res[ten] = L.cham(dec, y); res[ten]["dec"] = [int(v) for v in dec]
        if them:
            res[ten].update(them)
    ghi("E0", phat_lai_co_dinh(model, h1u, lg_c, ms))
    with torch.no_grad():                                               # luong CO DINH E0, nhan qua: W0 o 40 luot dau, SF_B sau do
        lg_e0 = torch.cat([lg_c[:N0], L.sau(model, h1u[N0:], ms)])
        lab_sf_dau = L.sau(model, h1u[:N0], ms).argmax(1)              # nhan cua SF_B cho 40 luot dau: SF_B co sau luot 40, khong dung nhan that
    luong_e0 = (lg_e0.argmax(1), L.do_tin(lg_e0))
    dec_goc, _ = L.phat_lai_mod(model, h1u, lg_c, ms, True, C)          # E1 bang CHINH ham cua job 18691
    dec, cf_e1, them = phat_lai_moc(model, h1u, lg_c, yt, ms, DW5, DB5, "W", "cu", "nhangia")
    assert np.array_equal(dec, dec_goc), "ham phat_lai_moc khong trung L.phat_lai_mod"
    ghi("E1", dec, them)
    luong_e1 = (torch.tensor(dec, device=dev, dtype=torch.long), cf_e1)
    dec_kiem, _, them_kiem = phat_lai_moc(model, h1u, lg_c, yt, ms, DW5, DB5, "W", "cu", "nhangia", ngoai=luong_e1)
    assert np.array_equal(dec_kiem, dec) and them_kiem["van_tay"] == them["van_tay"], "dung luong E1 lam ngoai khong tai lap E1"
    for ten, kg, nh, mm, kw in (("E1d", "W", "cu", "nhangia", {"lab_dau": lab_sf_dau, "ngoai": luong_e1}),
                                ("E1b", "W", "moi", "nhangia", {}), ("E2", "theta", "cu", "nhangia", {}), ("E2b", "theta", "moi", "nhangia", {}),
                                ("E3", "theta", "moi", "dotin", {}),
                                ("E1c", "W", "moi", "nhangia", {"ngoai": luong_e0}), ("E2c", "theta", "moi", "nhangia", {"ngoai": luong_e0})):
        dec, _, them = phat_lai_moc(model, h1u, lg_c, yt, ms, DW5, DB5, kg, nh, mm, **kw)
        ghi(ten, dec, them)
    assert res["E1c"]["van_tay"] == res["E2c"]["van_tay"], "E1c, E2c khong cung du lieu hoc"
    vt1, vtd = res["E1"]["van_tay"], res["E1d"]["van_tay"]             # E1, E1d: cung chi so luot o moi moc; nhan chi khac o 40 luot dau
    assert res["E1"]["so_luot_hoc"] == res["E1d"]["so_luot_hoc"] and len(vt1) == len(vtd)
    torch.manual_seed(seed + 7)                                         # doi chung toan mang nhu job 18691 (ban bat dropout, cung seed)
    ghi("all_seq", L.phat_lai_all(model, Xu, lg0, True))
    for ten, DWk, DBk, K in (("E4", DW5, DB5, K_CHINH), ("E5", DW11, DB11, K_RONG)):
        if ten == "E5":
            ghi("E5f", phat_lai_co_dinh(model, h1u, lg_c, gop(DWk, DBk, torch.full((K,), 1.0 / K, device=dev))))
        for hs in HE_SO_ETA:
            dec, them = phat_lai_eg(model, h1u, lg_c, DWk, DBk, hs)
            ghi(ten if hs == 1.0 else f"{ten}_eta{hs}", dec, them)
    d1 = cham_bac_nhat(model, h1u, mods, C, dev)
    top5b = torch.argsort(-d1, stable=True)[:K_CHINH]
    DW5b, DB5b = xep(mods, top5b)
    ghi("E6f", phat_lai_co_dinh(model, h1u, lg_c, gop(DW5b, DB5b, deu5)))
    dec, them = phat_lai_eg(model, h1u, lg_c, DW5b, DB5b, 1.0)
    ghi("E6", dec, them)
    res["bac_nhat"] = {"nguon_chon5": [src[int(j)] + 1 for j in top5b], "so_trung": len(set(top5.tolist()) & set(top5b.tolist())),
                       "q_chon5": [float(qs[int(j)]) for j in top5b],
                       "spearman": float(spearmanr(g.cpu().numpy(), d1.cpu().numpy()).correlation)}    # hang trung binh khi hoa diem
    for ten, DWk, DBk in (("O5", DW5, DB5), ("O11", DW11, DB11)):       # chan doan co nhan: tinh sau cung
        dec, them = khop_nhan_that(model, h1u, yt, lg_c, DWk, DBk)
        ghi(ten, dec, them)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="benchmark", choices=["benchmark", "beta", "lee"])
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--trongso", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--thiet-bi", default="cuda", choices=["cuda", "cpu"], help="cuda (nhu job 18691; mac dinh) hoac cpu (chi de thu)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    if a.thiet_bi == "cuda" and not torch.cuda.is_available():
        sys.exit("khong co CUDA")
    dev = torch.device(a.thiet_bi)
    print("thiet bi:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    ds = a.ds; DL.dang_ky(B, GD, ds); assert GD.NT == DL.SO_LOP[ds]; nl = DL.SO_LUOT[ds]; c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))
    if ds == "lee":                                                      # trong so phai cung bo nap va cung ban du lieu
        assert cfg.get("du_lieu_lee_sha256") == hashlib.sha256(open(DL.__file__, "rb").read()).hexdigest(), "bo nap lee khac luc huan luyen"
        assert cfg.get("lee_manifest_sha256") == DL.ma_manifest(), "ban du lieu lee khac luc huan luyen"
        CFG_NEN_LEE = dict(seed=20261107, epochs=1000, lo=100, nch=22, dropout_cuoi=0.95, loc="causal", subjects=0, ds="lee",
                           script_sha256="a1837d76c0a1dcef4a198a896b1bcf748852cffcb77c797fed407d36088657ca",
                           thichnghi_sha256="51040b5317ee2fe20ac7ca82f60d8713d1f9e20ad73eb9b7bd4abf3742afa257")
        for k_, v_ in CFG_NEN_LEE.items():                               # khoa du cau hinh mang nen
            assert cfg.get(k_) == v_, f"cau hinh mang nen {k_} = {cfg.get(k_)} khac {v_}"
    assert cfg["nch"] == NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg["subjects"] == 0 and cfg.get("ds", "benchmark") == ds
    D = {"X": [], "y": []}
    for s in range(1, nsub + 1):
        X, y, _ = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and X.shape[1] == 9 and np.isfinite(X).all() and np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, nl // GD.NT))
        D["X"].append(GD.tien_xu_ly(X, ds, "causal", ns, fs)); D["y"].append(y)
    for i in range(a.folds or nsub):
        assert os.path.exists(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")), i + 1
    t0 = time.time(); per = []
    for i in range(a.folds or nsub):
        per.append(chay_fold(i, D, a, dev, a.trongso))
        json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        print(f"fold {i+1} chay thong {time.time()-t0:.0f} s", flush=True)      # khong in do chinh xac trong luc chay
    mac = GD.dem_mac(None, 9, ns, nch=NCH)
    lan_nguoc = mac["fc"] + mac["l4"] + mac["l3"] + mac["l2"]
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    js = dict(seed=SEED, ds=ds, thiet_bi=str(dev), T=a.T, n0=N0, lo=LO, k_chinh=K_CHINH, k_rong=K_RONG, lr_theta_moc=LR_THETA_MOC, lr_theta_oracle=LR_THETA, buoc_oracle=BUOC_ORACLE, he_so_eta=HE_SO_ETA,
              buoc_nguon=L.BUOC_NGUON, buoc_cn=L.BUOC_CN, lr_mod=L.LR_MOD, nch=NCH, trongso=a.trongso, trongso_script_sha256=cfg["script_sha256"],
              root_du_lieu=c["root"],
              mac=dict(mang=mac, lan_truyen_nguoc_toi_l2=lan_nguoc, nen_moi_luot_K5=lan_nguoc + 2 * K_CHINH * (NCH * 9 + NCH),
                       cham_du_moi_luot=(nsub - 1) * (mac["tong"] - mac["l1"]), cham_bac_nhat_moi_luot=lan_nguoc + (nsub - 1) * (NCH * 9 + NCH),
                       ghi_chu="du toan phep nhan chinh, khong phai so do; lan truyen nguoc dung lai kich hoat cua luot suy luan"),
              script_sha256=sha(os.path.abspath(__file__)), b1moi_lai_sha256=sha(L.__file__), global_dnn_sha256=sha(GD.__file__),
              b1_diem_sha256=sha(B.__file__), du_lieu_lee_sha256=sha(DL.__file__),
              lee_manifest_sha256=(DL.ma_manifest() if ds == "lee" else None), trongso_config_sha256=sha(os.path.join(a.trongso, "config.json")),
              trongso_fold_sha256=[sha(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")) for i in range(a.folds or nsub)], tung_nguoi=per)
    json.dump(js, open(a.out, "w"), indent=1)
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
