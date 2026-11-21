#!/usr/bin/env python
"""PHEP THU 2 x 2: CHON NGUON de KHOI TAO mot mo-dun thich nghi nho tren bo trich dac trung CO DINH, roi cap nhat khong nhan.
Phep thu 2 x 2. Benchmark, LOSO, phat lai theo thu tu tep, n0 = 40.

Bo trich: Global DNN thu nho (nch 22, loc nhan qua) da huan luyen o lo 2 tren 34 nguoi nguon cua tung fold (results/b1v7_lo2/T??/foldNNN_trongso.pt), GIU CO DINH.
Mo-dun (hai kieu):
  A : nhan he so va cong do lech theo tung kenh dac trung truoc lop FC (2 x nch = 44 tham so); mo-dun chung = dong nhat (he so 1, do lech 0).
  B : so gia cua lop loc khong gian l2 (nch x C trong so + nch do lech = 220 tham so); mo-dun chung = 0.
Mo-dun cua tung nguoi nguon: tu mo-dun chung, hoc tren du lieu CO NHAN cua nguoi nguon do (bo trich co dinh, Adam lr 1e-3, BUOC_NGUON buoc ca lo, khong dropout).
Nguoi dung moi (khong nhan):
  luot 1..40 : quyet dinh bang mo-dun chung. Tich luy do tin (max - TB con lai)/sd con lai cua logit cua TUNG mo-dun nguon (chi phi tinh rieng).
  sau luot 40: mo-dun ca nhan = TB deu tham so cua K_SRC = 5 mo-dun nguon co do tin TB cao nhat.
  cap nhat   : tai cac moc 40, 60, 80, ...: BUOC_CN buoc Adam (lr 1e-3) entropy cheo voi nhan gia (= quyet dinh da phat) tren cac luot da qua co do tin
               >= phan vi 25 cua do tin 40 luot dau; chi tham so mo-dun. Moi luot duoc giai TRUOC khi vao cap nhat.
Bon o: CF (chung, co dinh = mang tinh), CU (chung + cap nhat), SF (tu nguon duoc chon, co dinh), SU (tu nguon duoc chon + cap nhat).
Doi chung ngoai bang: cap nhat TOAN MANG cung lich (lr 1e-4, L2 1e-3 tren trong so): all_seq = che do train (dropout bat, nhu all_pl cua lo 2);
all_seq_e = TAT dropout khi cap nhat (dropout 0,95 co the lam yeu doi chung). Adam tao lai o moi moc cho CA mo-dun va toan mang.
L3 xet voi doi chung toan mang co TB CAO HON trong hai ban.
Them 2026-11-14 (xac nhan tren BETA): --ds benchmark|beta; --xac-nhan: CHI tinh va xuat ung vien SF kieu B cung mang tinh (CF), khong tinh cac o khac.
--d7: cung mo-dun nguon da hoc, chi kieu B, chi luot 41..het, khong cap nhat; doi chung cho KTN:
  ngau5 = TB tham so 5 nguon ngau nhien (R_D7 = 20 lan, rng(seed + 991) moi fold, ket qua TB qua lan); kem5 = TB tham so 5 nguon co do tin THAP nhat;
  tatca = TB tham so moi nguon; ens5 = TB LOGIT (du doan) cua 5 nguon duoc chon, kieu Ensemble-DNN (5 mang, vuot tran).
Cham: luot 41..het (chinh), tung doan 40 luot, ca phien. Nhan that chi de cham va de bao chat luong mo-dun nguon.
"""
import argparse, copy, json, os, sys, time, hashlib
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402
import du_lieu_eld as DE     # noqa: E402

SEED = int(os.environ.get("B2_SEED", "20261107"))   # seed huan luyen va phat lai; mac dinh nhu bai
N0, K_SRC, LO = 40, 5, 20
R_D7 = 20
BUOC_NGUON, BUOC_CN = 100, 10
LR_MOD, LR_ALL = 1e-3, 1e-4
NCH = 22


def mod_chung(kieu, C, dev):
    if kieu == "A":
        return {"g": torch.ones(NCH, device=dev), "b": torch.zeros(NCH, device=dev)}
    return {"dW": torch.zeros(NCH, 1, C, 1, device=dev), "db": torch.zeros(NCH, device=dev)}


def truoc(model, x):
    """Phan khong phu thuoc mo-dun: A -> dac trung sau l4; B -> dau ra l1."""
    return model.l1(x)


def sau(model, h1, mod):
    """h1 = dau ra l1 (B, 1, C, ns). Khong dropout (bo trich co dinh, che do eval)."""
    if "dW" in mod:
        h = Fn.conv2d(h1, model.l2.weight + mod["dW"], model.l2.bias + mod["db"])
    else:
        h = model.l2(h1)
    h = model.l4(torch.relu(model.l3(h)))
    if "g" in mod:
        h = h * mod["g"][None, :, None, None] + mod["b"][None, :, None, None]
    return model.fc(h.flatten(1))


def hoc(model, h1, lab, mod0, buoc, lr):
    mod = {k: v.clone().requires_grad_(True) for k, v in mod0.items()}
    opt = torch.optim.Adam(list(mod.values()), lr=lr)
    for _ in range(buoc):
        opt.zero_grad(set_to_none=True)
        Fn.cross_entropy(sau(model, h1, mod), lab).backward(); opt.step()
    return {k: v.detach() for k, v in mod.items()}


def do_tin(lg):
    """lg (..., N) -> (max - TB con lai) / sd con lai."""
    s, _ = torch.sort(lg, dim=-1); o = s[..., :-1]
    return (s[..., -1] - o.mean(-1)) / (o.std(-1, unbiased=False) + 1e-12)


def cham(dec, y):
    ok = (dec == y)
    e = dict(sau=float(ok[N0:].mean()), phien=float(ok.mean()))
    e["doan"] = [float(ok[a:a + 40].mean()) for a in range(N0, len(y), 40)]
    return e


def phat_lai_mod(model, h1, lg_c, mod0, cap_nhat, C):
    """Tra quyet dinh (n,). lg_c: logit mo-dun chung (n, N). mod0: mo-dun dung tu luot N0 (None = chung)."""
    n = h1.shape[0]; dec = torch.empty(n, dtype=torch.long, device=h1.device)
    dec[:N0] = lg_c[:N0].argmax(1)
    cf = torch.empty(n, device=h1.device); cf[:N0] = do_tin(lg_c[:N0])
    nguong = torch.quantile(cf[:N0], 0.25)
    mod = {k: v.clone() for k, v in mod0.items()}
    so_lan = 0
    for a in range(N0, n, LO):
        if cap_nhat:
            idx = torch.where(cf[:a] >= nguong)[0]
            mod = hoc(model, h1[idx], dec[idx], mod, BUOC_CN, LR_MOD); so_lan += 1
        with torch.no_grad():
            lg = sau(model, h1[a:a + LO], mod)
        dec[a:a + LO] = lg.argmax(1); cf[a:a + LO] = do_tin(lg)
    return dec.cpu().numpy(), so_lan


def phat_lai_all(model0, X, lg_c, bat_dropout):
    """Doi chung: cap nhat toan mang cung lich, nhu all_pl cua lo 2 (che do train khi cap nhat)."""
    m = copy.deepcopy(model0)
    for p in m.parameters():
        p.requires_grad_(True)
    ps = list(m.named_parameters())
    n = X.shape[0]; dec = torch.empty(n, dtype=torch.long, device=X.device)
    dec[:N0] = lg_c[:N0].argmax(1)
    cf = torch.empty(n, device=X.device); cf[:N0] = do_tin(lg_c[:N0])
    nguong = torch.quantile(cf[:N0], 0.25)
    for a in range(N0, n, LO):
        idx = torch.where(cf[:a] >= nguong)[0]
        opt = torch.optim.Adam([{"params": [p for k, p in ps if k.endswith("weight")], "weight_decay": 1e-3},
                                {"params": [p for k, p in ps if k.endswith("bias")], "weight_decay": 0.0}], lr=LR_ALL)   # tao lai moi moc, nhu mo-dun
        m.train() if bat_dropout else m.eval()
        for _ in range(BUOC_CN):
            opt.zero_grad(set_to_none=True)
            Fn.cross_entropy(m(X[idx]), dec[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lg = m(X[a:a + LO])
        dec[a:a + LO] = lg.argmax(1); cf[a:a + LO] = do_tin(lg)
    return dec.cpu().numpy()


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
    Xu = torch.tensor(D["X"][i], device=dev); y = D["y"][i]
    with torch.no_grad():
        h1u = truoc(model, Xu)
        lg0 = model(Xu)
        for kk_ in ("A", "B"):                                                   # mo-dun chung phai tuong duong forward eval (ca hai kieu)
            assert torch.allclose(lg0, sau(model, h1u, mod_chung(kk_, C, dev)), atol=1e-4), kk_
    res = {"fold": i + 1, "seed": seed}
    for kieu in (("B",) if a.xac_nhan else ("A", "B")):
        mc = mod_chung(kieu, C, dev)
        with torch.no_grad():
            lg_c = sau(model, h1u, mc)
        # mo-dun cua tung nguoi nguon (du lieu co nhan cua NGUON; nguoi thu i khong tham gia)
        mods = []
        for s in src:
            Xs = torch.tensor(D["X"][s], device=dev); ys = torch.tensor(D["y"][s], device=dev, dtype=torch.long)
            with torch.no_grad():
                h1s = truoc(model, Xs)
            mods.append(hoc(model, h1s, ys, mc, BUOC_NGUON, LR_MOD))
        with torch.no_grad():
            lgs = torch.stack([sau(model, h1u, m) for m in mods])            # (S, n, N)
        g = do_tin(lgs[:, :N0]).mean(1)                                       # (S,) khong dung nhan
        top = torch.argsort(-g, stable=True)[:K_SRC]
        ms = {k: torch.stack([mods[int(j)][k] for j in top]).mean(0) for k in mc}
        e = {"nguon_chon": [src[int(j)] + 1 for j in top]}
        qs = (lgs[:, N0:].argmax(2).cpu().numpy() == y[None, N0:]).mean(1)    # chat luong tung mo-dun nguon (nhan CHI de bao cao)
        e["q_nguon_tb"] = float(qs.mean()); e["q_nguon_chon"] = float(qs[top.cpu().numpy()].mean()); e["q_nguon_tot_nhat"] = float(qs.max())
        for ten, m0, cn in ((("CF", mc, False), ("SF", ms, False)) if a.xac_nhan else (("CF", mc, False), ("CU", mc, True), ("SF", ms, False), ("SU", ms, True))):
            dec, so_lan = phat_lai_mod(model, h1u, lg_c, m0, cn, C)
            e[ten] = cham(dec, y); e[ten]["so_lan_cap_nhat"] = so_lan
        if a.d7 and kieu == "B":
            ys_ = y[N0:]
            hit = lambda lg_: float((lg_.argmax(1).cpu().numpy() == ys_).mean())
            tb_mod = lambda js: {k: torch.stack([mods[int(j)][k] for j in js]).mean(0) for k in mc}
            with torch.no_grad():
                e["d7_ktn"] = hit(sau(model, h1u[N0:], ms))
                rg = np.random.default_rng(seed + 991); S_ = len(mods)
                e["d7_ngau5_lan"] = [hit(sau(model, h1u[N0:], tb_mod(rg.choice(S_, K_SRC, replace=False)))) for _ in range(R_D7)]
                e["d7_ngau5"] = float(np.mean(e["d7_ngau5_lan"]))
                e["d7_kem5"] = hit(sau(model, h1u[N0:], tb_mod(torch.argsort(g, stable=True)[:K_SRC])))
                e["d7_tatca"] = hit(sau(model, h1u[N0:], tb_mod(range(S_))))
                e["d7_ens5"] = hit(lgs[top][:, N0:].mean(0))
        res[kieu] = e
    res["tinh"] = cham(lg0.argmax(1).cpu().numpy(), y)
    if a.xac_nhan:
        return res
    torch.manual_seed(seed + 7)
    res["all_seq"] = cham(phat_lai_all(model, Xu, lg0, True), y)
    torch.manual_seed(seed + 7)
    res["all_seq_e"] = cham(phat_lai_all(model, Xu, lg0, False), y)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="benchmark", choices=["benchmark", "beta", "eld"])
    ap.add_argument("--xac-nhan", action="store_true")
    ap.add_argument("--d7", action="store_true")
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--trongso", required=True, help="thu muc trong so lo 2 cua cua so T")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=0, help="chi chay so fold dau (thu); 0 = tat ca")
    ap.add_argument("--im-lang", action="store_true")
    ap.add_argument("--cho-phep-cpu", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    if not torch.cuda.is_available() and not a.cho_phep_cpu:
        sys.exit("khong co CUDA")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    ds = a.ds; DE.dang_ky(B, GD, ds); assert GD.NT == DE.SO_LOP[ds]; nl = DE.SO_LUOT[ds]; c = B.DS[ds]; fs = c["fs"]; ns = int(round(a.T * fs)); nsub = c["n"]
    cfg = json.load(open(os.path.join(a.trongso, "config.json")))
    assert cfg["nch"] == NCH and cfg["loc"] == "causal" and abs(cfg["T"] - a.T) < 1e-9 and cfg["subjects"] == 0 and cfg.get("ds", "benchmark") == ds
    D = {"X": [], "y": []}
    for s in range(1, nsub + 1):
        X, y, _ = B.load_subject(ds, s, B.CH9)
        assert X.shape[0] == nl and X.shape[1] == 9 and np.isfinite(X).all() and np.array_equal(np.bincount(y, minlength=GD.NT), np.full(GD.NT, nl // GD.NT))
        D["X"].append(GD.tien_xu_ly(X, ds, "causal", ns, fs)); D["y"].append(y)
    t0 = time.time(); per = []
    for i in range(a.folds or nsub):                                             # du 35 tep trong so truoc khi chay
        assert os.path.exists(os.path.join(a.trongso, f"fold{i+1:03d}_trongso.pt")), i + 1
    for i in range(a.folds or nsub):
        per.append(chay_fold(i, D, a, dev, a.trongso))
        if not a.im_lang:                                                        # luu tam tung fold de khong mat khi het gio
            json.dump(per, open(a.out + ".dang-chay.tmp", "w"))
        if not a.im_lang and not a.xac_nhan:
            r = per[-1]
            print(f"fold {i+1}: tinh {r['tinh']['sau']:.3f} all_seq {r['all_seq']['sau']:.3f} all_seq_e {r['all_seq_e']['sau']:.3f} | " +
                  " | ".join(f"{k} " + " ".join(f"{o} {r[k][o]['sau']:.3f}" for o in ("CF", "CU", "SF", "SU")) for k in ("A", "B")) + f" | {time.time()-t0:.0f} s", flush=True)
        else:
            print(f"fold {i+1} chay thong {time.time()-t0:.0f} s", flush=True)
    n2 = ns // 2; mac = GD.dem_mac(None, 9, ns, nch=NCH)
    js = dict(seed=SEED, ds=ds, xac_nhan=a.xac_nhan, d7=a.d7, r_d7=R_D7, T=a.T, n0=N0, k_src=K_SRC, lo=LO, buoc_nguon=BUOC_NGUON, buoc_cn=BUOC_CN, lr_mod=LR_MOD, lr_all=LR_ALL, nch=NCH,
              trongso=a.trongso, trongso_script_sha256=cfg["script_sha256"],
              mac=dict(mang=mac, modun_A_quyet_dinh=NCH * n2, so_tham_so_A=2 * NCH, so_tham_so_B=NCH * 9 + NCH,
                       cham_nguon_moi_luot_A_ma_nay=(nsub - 1) * (mac["tong"] - mac["l1"] + NCH * n2), cham_nguon_moi_luot_A_neu_luu_dac_trung=(nsub - 1) * (NCH * n2 + mac["fc"]),
                       cham_nguon_moi_luot_B=(nsub - 1) * (mac["tong"] - mac["l1"]),
                       ghi_chu="du toan phep nhan chinh; chi 40 luot dau can cham mo-dun nguon, ma mo phong cham ca 240 luot de bao chat luong nguon"),
              script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
              global_dnn_sha256=hashlib.sha256(open(GD.__file__, "rb").read()).hexdigest(), tung_nguoi=per)
    if not a.im_lang:
        json.dump(js, open(a.out, "w"), indent=1)
    print("xong", time.time() - t0)


if __name__ == "__main__":
    main()
