#!/usr/bin/env python
"""BAN SUY LUAN THAT cua kh_deu (B1 moi): xu ly TUNG LUOT theo luong, chi tinh dai va nguon dang dung, co BO DEM phep nhan.
Khac ma mo phong (b1moi_buoc0b.py tinh moi nguon 5 dai cho moi luot): o day
  pha khoi dong (luot 1..n0): S nguon x nb dai; tich luy thong ke tin cay va tong phep chieu theo nhan gia (chi phep cong);
  chon mot lan sau luot n0: tuong quan mau nhan gia - mau nguon, z-diem, lay K nguon;
  pha sau: K nguon x 5 dai.
Hai che do:
  --kiem <ds>   : (HPC) dung kho nguon LOSO tu du lieu that, phat lai tung luot, so do chinh xac tung nguoi voi JSON mo phong (phai trung).
  --do-tre      : (thiet bi nhung) kho nguon ngau nhien cung kich thuoc, do thoi gian moi quyet dinh va lan chon nguon; khong can du lieu.
Bo dem (moi quyet dinh): chieu = C*ns moi (nguon, dai); tuong_quan = N*ns; chuan = ns; khac = trong so dai, chia, can (dem rieng).
"""
import argparse, json, os, sys, time, hashlib
import numpy as np

SEED = 20261107
FBW = np.array([(m + 1) ** -1.25 + 0.25 for m in range(5)])
NB = 5


class KhDeu:
    def __init__(self, W, ts, n0=40, K=11, cap=1000000):
        """W (S, NB, C) bo loc tung nguon; ts (S, N, NB, ns) mau da chieu, da tru TB theo thoi gian."""
        self.W = W; self.ts = ts
        self.S, _, self.C = W.shape; self.N = ts.shape[1]; self.ns = ts.shape[3]
        self.tn = np.sqrt((ts ** 2).sum(-1))                                  # chuan mau nguon LUU SAN (S, N, NB)
        u = self.ns * (self.C + self.N + 1)
        self.nb = int(min(2, max(1, cap // (u * self.S))))
        self.wk = FBW[:self.nb] / FBW[:self.nb].sum(); self.w5 = FBW / FBW.sum()
        self.n0 = n0; self.K = min(K, self.S)
        self.i = 0; self.sel = None
        self.g_rcs = np.zeros(self.S); self.acc = np.zeros((self.N, self.S, self.nb, self.ns)); self.cnt = np.zeros(self.N, int)
        self.mac = dict(chieu=0, tuong_quan=0, chuan=0, khac=0); self.mac_chon = None

    def _diem(self, x, src, nb):
        """x (NB, C, ns) da tru TB theo thoi gian. Tra z (s, nb, ns) va r (s, N, nb)."""
        W = self.W[src, :nb] if src is not None else self.W[:, :nb]
        ts = self.ts[src, :, :nb] if src is not None else self.ts[:, :, :nb]
        tn = self.tn[src, :, :nb] if src is not None else self.tn[:, :, :nb]
        z = np.einsum("sbc,bct->sbt", W, x[:nb])
        num = np.einsum("sbt,snbt->snb", z, ts)
        zn = np.sqrt((z ** 2).sum(-1))
        r = num / (zn[:, None, :] * tn + 1e-150)
        s = z.shape[0]
        self.mac["chieu"] += s * nb * self.C * self.ns; self.mac["tuong_quan"] += s * nb * self.N * self.ns
        self.mac["chuan"] += s * nb * self.ns; self.mac["khac"] += s * nb * (1 + 2 * self.N)      # can, nhan chuan, chia
        return z, r

    def quyet_dinh(self, x):
        self.mac = dict(chieu=0, tuong_quan=0, chuan=0, khac=0)
        if self.sel is None:
            z, r = self._diem(x, None, self.nb)
            s2 = r @ self.wk                                                  # (S, N)
            self.mac["khac"] += self.S * self.N * self.nb
            d = int(s2.mean(0).argmax())
            # tich luy khong nhan (ngoai duong quyet dinh): do tin va tong phep chieu theo nhan gia
            srt = np.sort(s2, 1); oth = srt[:, :-1]
            self.g_rcs += (srt[:, -1] - oth.mean(1)) / (oth.std(1) + 1e-12)
            self.acc[d] += z; self.cnt[d] += 1
            self.i += 1
            if self.i == self.n0:
                self._chon()
            return d
        z, r = self._diem(x, self.sel, NB)
        self.mac["khac"] += self.K * self.N * NB
        self.i += 1
        return int((r @ self.w5).mean(0).argmax())

    def _chon(self):
        co = np.where(self.cnt > 0)[0]
        g = np.zeros(self.S); m_tq = 0; m_ch = 0
        for k in co:
            m = self.acc[k] / self.cnt[k]
            m = m - m.mean(-1, keepdims=True)                                 # (S, nb, ns)
            t = self.ts[:, k, :self.nb]
            c = (m * t).sum(-1) / np.sqrt((m * m).sum(-1) * (t * t).sum(-1) + 1e-300)
            g += self.cnt[k] * (c @ self.wk)
            m_tq += self.S * self.nb * self.ns; m_ch += self.S * self.nb * self.ns      # chuan mau nguon da luu san
        g_pl = g / self.cnt[co].sum(); g_rcs = self.g_rcs / self.n0
        zc = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        self.sel = np.sort(np.argsort(-(zc(g_pl) + zc(g_rcs)), kind="stable")[:self.K])
        self.mac_chon = dict(tuong_quan=m_tq, chuan=m_ch, so_lop_co_nhan_gia=int(len(co)))


def kiem(a):
    """So voi mo phong: do chinh xac tung nguoi cua kh_deu (luot sau n0) phai trung JSON."""
    from scipy.linalg import eigh
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import b1_diem as B
    ref = json.load(open(a.json, encoding="utf-8"))
    c = B.DS[a.kiem]; fs = c["fs"]; sos = B.filterbank(fs); nsub = c["n"]
    P, Y = [], []
    for s in range(1, nsub + 1):
        X, y, _ = B.load_subject(a.kiem, s, B.CH9)
        P.append(B.preprocess(X, a.kiem, sos, int(round(1.0 * fs))).astype(np.float32)); Y.append(y)
    out = {}
    for T in (0.6, 1.0):
        ns = int(round(T * fs)); NT = B.NTGT
        X = [B.center(p[..., :ns].astype(np.float64)) for p in P]
        M = np.stack([np.stack([x[Y[s] == k].mean(0) for k in range(NT)]) for s, x in enumerate(X)])
        G = np.einsum("snbct,snbdt->sbcd", M, M)
        R = np.stack([np.einsum("nbct,nbdt->bcd", x - M[s][Y[s]], x - M[s][Y[s]]) for s, x in enumerate(X)])

        def top(Sm, Q):
            C = Sm.shape[0]
            return eigh((Sm + Sm.T) / 2, Q + 1e-6 * np.trace(Q) / C * np.eye(C))[1][:, -1]
        Ws = np.stack([np.stack([top(G[s, b], R[s, b]) for b in range(NB)]) for s in range(nsub)])
        ts = B.center(np.einsum("sbc,snbct->snbt", Ws, M))
        lech = []; mx = dict(khoi_dong=0, sau=0); chon = None
        for u in range(nsub):
            src = [s for s in range(nsub) if s != u]
            kd = KhDeu(Ws[src], ts[src])
            dec = np.empty(len(Y[u]), int)
            for i in range(len(dec)):
                dec[i] = kd.quyet_dinh(X[u][i])
                tong = kd.mac["chieu"] + kd.mac["tuong_quan"] + kd.mac["chuan"]
                key = "khoi_dong" if i < 40 else "sau"; mx[key] = max(mx[key], tong)
            chon = kd.mac_chon
            acc = float((dec[40:] == Y[u][40:]).mean())
            lech.append(abs(acc - ref["ket_qua"][str(T)]["tung_nguoi"][u]["40"]["kh_deu"]))
        out[str(T)] = dict(lech_max=float(np.max(lech)), so_nguoi_lech=int(np.sum(np.array(lech) > 1e-12)), mac_dem_max=mx, mac_chon_mot_lan=chon)
        print(T, json.dumps(out[str(T)]), flush=True)
    json.dump(dict(ds=a.kiem, ket_qua=out, script_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()), open(a.out, "w"), indent=1)


def do_tre(a):
    rng = np.random.default_rng(SEED)
    res = {}
    for ten, S, N, C in (("benchmark", 34, 40, 9), ("beta", 69, 40, 9), ("eldbeta", 99, 9, 9), ("lee2019", 53, 4, 9)):
        for T in (0.6, 1.0):
            ns = int(round(T * 250))
            W = rng.standard_normal((S, NB, C)); ts = rng.standard_normal((S, N, NB, ns)); ts -= ts.mean(-1, keepdims=True)
            kd = KhDeu(W, ts)
            xs = rng.standard_normal((40 + a.lap, NB, C, ns)); xs -= xs.mean(-1, keepdims=True)
            t_kd, t_sau = [], []; t_chon = None
            for i in range(len(xs)):
                t0 = time.perf_counter(); kd.quyet_dinh(xs[i]); dt = time.perf_counter() - t0
                if i < 39:
                    t_kd.append(dt); m_kd = dict(kd.mac)
                elif i == 39:
                    t_chon = dt
                else:
                    t_sau.append(dt); m_sau = dict(kd.mac)
            q = lambda v: dict(tb_ms=float(1e3 * np.mean(v)), trung_vi_ms=float(1e3 * np.median(v)), p95_ms=float(1e3 * np.percentile(v, 95)), max_ms=float(1e3 * np.max(v)))
            res[f"{ten}_T{T}"] = dict(S=S, N=N, C=C, ns=ns, so_dai_khoi_dong=kd.nb, khoi_dong=q(t_kd[5:]), sau=q(t_sau[5:]), luot_chon_nguon_ms=float(1e3 * t_chon),
                                      mac_khoi_dong=m_kd, mac_sau=m_sau, mac_chon=kd.mac_chon)
            print(ten, T, json.dumps(res[f"{ten}_T{T}"]), flush=True)
    json.dump(dict(che_do="do-tre", lap=a.lap, luong=os.environ.get("OMP_NUM_THREADS", ""), numpy=np.__version__, ket_qua=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kiem", default="", choices=["", "benchmark", "beta"])
    ap.add_argument("--json", default="")
    ap.add_argument("--do-tre", action="store_true")
    ap.add_argument("--lap", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("tep ket qua da co: khong ghi de")
    kiem(a) if a.kiem else do_tre(a)
