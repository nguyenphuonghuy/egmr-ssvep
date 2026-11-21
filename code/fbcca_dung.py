#!/usr/bin/env python
"""FBCCA DUNG CONG THUC Chen 2015 (pt. 6: rho_c = sum_m w_m (rho_{m,c})^2, w_m = m^-1,25 + 0,25), lam doi chung khong huan luyen cho ban thao B1.
Sua loi (2026-11-18): FBCCA cu lay tu rcs_tai_hien.py cong rho CHUA binh phuong. Khong sua tep cu.
Cung tien xu ly voi KTN-R/b1_diem: tru TB doan truoc kich thich, loc nhan qua (sosfilt) tu mau 0, cat [onset + tre, + ns); 5 dai [8m, 88] Hz
Chebyshev I (cheb1ord, gon 0,5 dB) cua b1_diem.filterbank; tham chieu sin/cos 5 hai (b1_diem.refs). Cham luot 41..het (chi so 0-based >= 40).
Bo: benchmark, beta, eld, dong, nak (nak them 2026-11-18, chay job rieng) (dang ky bo nap nhu ktnr_buoc1.py). Kenh: CH9 (dong: 8 kenh cua bo nap).
Dung: python fbcca_dung.py --out <json>
"""
import argparse, json, os, sys, time, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1_diem as B          # noqa: E402
import global_dnn as GD      # noqa: E402
import du_lieu_eld as DE     # noqa: E402
import du_lieu_dong as DG    # noqa: E402
import du_lieu_nak as DN     # noqa: E402

N0 = 40
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bo", nargs="*", default=["benchmark", "beta", "eld", "dong"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    me = os.path.abspath(__file__)
    out = dict(seed=None, phuong_phap="FBCCA Chen 2015 pt. 6 (rho binh phuong, trong so m^-1.25+0.25), 5 dai, 5 hai, loc nhan qua, cham luot 41..het",
               script_sha256=sha(me), b1_diem_sha256=sha(os.path.join(os.path.dirname(me), "b1_diem.py")), bo={})
    for ds in a.bo:
        DE.dang_ky(B, GD, ds); DN.dang_ky(B, GD, ds); DG.dang_ky(B, GD, ds)
        c = B.DS[ds]; fs = c["fs"]; sos = B.filterbank(fs); out["bo"][ds] = {}
        for T in (1.0, 0.6):
            ns = int(round(T * fs)); t0 = time.time(); per = []
            for s in range(1, c["n"] + 1):
                X, y, fr = B.load_subject(ds, s, B.CH9)
                Xp = B.preprocess(X, ds, sos, ns)                              # (luot, 5, C, ns)
                Qy = B.refs(fr, fs, ns)                                         # (N, ns, 2H)
                n, nb, C, _ = Xp.shape; N = len(fr)
                sc = np.zeros((n, N))
                for i in range(n):
                    for m in range(nb):
                        qx = np.linalg.qr(B.center(Xp[i, m]).T)[0]
                        r = np.linalg.svd(np.einsum("tc,ktj->kcj", qx, Qy), compute_uv=False)[:, 0]
                        sc[i] += B.FBW[m] * r ** 2
                ok = (sc.argmax(1) == y)
                per.append(dict(nguoi=s, sau=float(ok[N0:].mean()), phien=float(ok.mean()), n_cham=int(n - N0)))
            mac = 5 * int(B.macs("fbcca", C, ns, n=N))
            out["bo"][ds][str(T)] = dict(tb_sau=100 * float(np.mean([p["sau"] for p in per])), so_nguoi=len(per), C=C, N=N,
                                        mac_5dai=mac, tung_nguoi=per)
            print(ds, T, "FBCCA sau %.2f %% (%d nguoi, C=%d, N=%d, MAC %d) %.0f s" % (out["bo"][ds][str(T)]["tb_sau"], len(per), C, N, mac, time.time() - t0), flush=True)
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
