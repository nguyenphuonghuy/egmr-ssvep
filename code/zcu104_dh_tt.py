#!/usr/bin/env python
"""Do DH-TT (Duan va Hao 2026, ban chuyen) tren PS cua ZCU104.
In MOT dong JSON. Can cung thu muc: global_dnn.py, b1_diem.py, zcu104_b1gop_eg.py, dh_tt_loi.py;
du lieu du_lieu_ben/benchmark_S1.npz. Cung cach do voi zcu104_b2e4r.py (EGM-R): trong so ngau nhien (chi do thoi gian, bo nho), lap lai
240 luot tho cua nguoi 1, 20 luot khoi dong may + n luot do, mot luong la cau hinh quyet dinh.
Mang: GlobalDNN thu nho (nch 22, global_dnn.py), che do eval (tat dropout) nhu phat_lai_dh. R_train: TB hiep phuong sai dai 1 cua 40 luot dau
(tinh ngoai vong do; chi lam diem xuat phat cho EA, khong anh huong thoi gian).
Moi luot (dong ho bat dau TRUOC GD.tien_xu_ly, gom thiet ke bo loc trong loi goi):
  bank loc  : tien xu ly nhan qua 3 dai                                                                        [t_bank]
  EA        : cov dai 1, R(i) = (n R + w cov)/(n + w), eigh 9x9, R^{-1/2} nhan 3 dai                           [t_ea]
  quyet dinh: luot xuoi giu do thi (moi tham so la la co grad) + argmax                                       [t_quyet_dinh = bank + EA + xuoi]
  nen       : mat mat DH (entropy + lambda R_freq, tau 2), lan truyen nguoc toan mang, SGD mot buoc, cap nhat F   [t_nen]
Bo nho them: storage duy nhat cua tensor autograd giu lai (loai storage tham so) + gradient moi tham so + trang thai EA (R, P) va F.
--kiem: doi chieu chuoi quyet dinh va tham so cuoi voi dh_tt_loi.phat_lai_dh tren cung du lieu, cung trong so (cong dung cua duong do).
"""
import argparse, io, json, os, sys, time, contextlib, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FS, NSB = 250, 3


def q(v):
    v = np.asarray(v, float)
    return dict(trung_vi_ms=1e3 * float(np.median(v)), p95_ms=1e3 * float(np.percentile(v, 95)), max_ms=1e3 * float(v.max()),
                tb_ms=1e3 * float(v.mean()), n=int(len(v)), vuot_50ms=float((v > 0.05).mean()), vuot_500ms=float((v > 0.5).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--luong", type=int, default=1)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--khoi-dong", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261107); ap.add_argument("--tien-trinh", type=int, default=0)
    ap.add_argument("--du-lieu", default="du_lieu_ben"); ap.add_argument("--kiem", action="store_true")
    a = ap.parse_args()
    import global_dnn as GD
    import zcu104_b1gop_eg as Z
    import dh_tt_loi as DH
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed)
    zf = np.load(os.path.join(a.du_lieu, "benchmark_S1.npz")); Xr = zf["X"]
    ns, C, N, nch = int(round(a.T * FS)), 9, 40, 22
    assert Xr.shape[1] == C, Xr.shape
    m = GD.GlobalDNN(C, ns, nclass=N, nch=nch).eval()
    m0 = copy.deepcopy(m)
    ps = list(m.parameters())
    for p in ps:
        p.requires_grad_(True)
    n_tham_so = sum(p.numel() for p in ps)
    dem = Z.DemLuu([p.untyped_storage().data_ptr() for p in ps])
    Xp40 = GD.tien_xu_ly(Xr[:40], "benchmark", "causal", ns, FS)            # ngoai vong do
    R_train = DH.r_nguon(Xp40)
    ea = DH.EATrucTuyen(R_train)
    F = torch.zeros(N, dtype=torch.float64)
    tb, te, tx, tq, tn, tt = [], [], [], [], [], []
    kich_hoat, kich_hoat_tho, tam, decs, Xdo = 0, 0, 0, [], []
    P = None
    for i in range(a.khoi_dong + a.n):
        xr = Xr[i % len(Xr)][None]
        t0 = time.perf_counter()
        x = GD.tien_xu_ly(xr, "benchmark", "causal", ns, FS)[0]
        t1 = time.perf_counter()
        P = ea.cap_nhat(x[0]); xa = DH.can_chinh(x, P)
        t2 = time.perf_counter()
        xt = torch.from_numpy(np.ascontiguousarray(xa[None]))
        for p in ps:
            p.grad = None
        dem.reset()
        with torch.autograd.graph.saved_tensors_hooks(dem.goi, lambda t: t):
            z = m(xt)
            d = int(z.detach().argmax(1)[0])
            t3 = time.perf_counter()
            L, logq = DH.mat_mat(z, F)
        L.backward()
        b, _ = DH.buoc_sgd(ps, DH.ALPHA_BAI)
        F = DH.cap_nhat_dem(F, logq)
        t4 = time.perf_counter()
        if b:
            raise FloatingPointError(f"gradient khong huu han o luot {i}")
        if a.kiem:
            decs.append(d); Xdo.append(x)
        if i >= a.khoi_dong:
            tb.append(t1 - t0); te.append(t2 - t1); tx.append(t3 - t2); tq.append(t3 - t0); tn.append(t4 - t3); tt.append(t4 - t0)
            kich_hoat = max(kich_hoat, dem.duy_nhat); kich_hoat_tho = max(kich_hoat_tho, dem.tho)
            tam = max(tam, Z.nb(*[p.grad for p in ps]))
    kiem = None
    if a.kiem:                                                             # cong: cung du lieu da loc, cung trong so dau, cung R_train
        r = DH.phat_lai_dh(m0, np.stack(Xdo), DH.ALPHA_BAI, "cpu", R_train=R_train)
        lech = max(float((p1.detach() - p2.detach()).abs().max()) for p1, p2 in zip(ps, r["model"].parameters()))
        kiem = dict(quyet_dinh_trung=float(np.mean(np.array(decs) == r["dec"])), lech_tham_so_max=lech,
                    F_trung=bool(np.array_equal(F.numpy(), r["F"])), dat=bool(np.array_equal(np.array(decs), r["dec"]) and lech < 1e-6
                                                                            and np.array_equal(F.numpy(), r["F"])))
    nb_np = lambda *v: sum(x.nbytes for x in v)
    trang_thai = nb_np(ea.R, P) + Z.nb(F)
    n2 = ns // 2
    lop = dict(l1=NSB * C * ns, l2=nch * C * ns, l3=nch * n2 * nch * 2, l4=nch * n2 * nch * 10, fc=nch * n2 * N)
    mac_mang = sum(lop.values())
    mac_ea = C * C * ns + NSB * C * C * ns                                   # cov dai 1 + can chinh 3 dai (eigh 9x9 khong dem)
    mac_nguoc = 2 * (lop["l2"] + lop["l3"] + lop["l4"] + lop["fc"]) + lop["l1"]  # grad dau vao + grad trong so; l1 chi grad trong so
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    print(json.dumps(dict(phan="dh_tt", T=a.T, luong=a.luong, tien_trinh=a.tien_trinh, seed=a.seed, n=a.n, khoi_dong=a.khoi_dong,
                          alpha=DH.ALPHA_BAI, n_tham_so=n_tham_so, torch=torch.__version__, numpy=np.__version__, torch_luong=torch.get_num_threads(),
                          env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                          ket_qua=dict(bank_loc=q(tb), ea=q(te), xuoi=q(tx), quyet_dinh_tu_tho=q(tq), nen=q(tn), tong_cung_luot=q(tt)),
                          mac=dict(suy_luan_mang=mac_mang, ea=mac_ea, quyet_dinh=mac_mang + mac_ea, nguoc=mac_nguoc, sgd=n_tham_so,
                                   nen=mac_nguoc + n_tham_so, ghi_chu="du toan theo cong thuc lop; khong gom bank loc, eigh 9x9, mat mat"),
                          byte=dict(kich_hoat_duy_nhat=kich_hoat, kich_hoat_tho=kich_hoat_tho, gradient=tam, trang_thai=trang_thai,
                                    them_thich_nghi=kich_hoat + tam + trang_thai),
                          kiem=kiem, vmhwm_kb=Z.vmhwm_kb(), np_config=buf.getvalue()[-800:])))
    if kiem is not None and not kiem["dat"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
