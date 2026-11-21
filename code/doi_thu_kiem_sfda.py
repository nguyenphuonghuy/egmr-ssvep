#!/usr/bin/env python
"""Kiem lop (a) do dung ban chuyen SFDA. Can torch; chay CPU.
K1 silhouette cosine (quy uoc MATLAB) so voi vong lap doc lap, co cum mot phan tu.
K2 lang gieng, mat na, trong so so voi ban dich tung dong neighbors_calculation.m; phan vi du khong co do sut (40 tin hieu cos lech pha 0,001 rad).
K3 mat mat: ca tinh tay (N = 4, N+ = 2, lam = 0,6, xac suat 1/2: lam * sl = 0,415888), so hang cuc bo voi mau so N va N + N_lop, gradient
   so voi sai phan huu han; tap loc rong.
K4 luong dieu khien: chuoi tu choi/nhan va mau so [80, 80, 40, ...], ba lan tu choi [80, 80, 80] va khong tra mang.
K5 quy tac du phong (viii) cua ban truc tuyen qua main-loop thu nho. Thoat ma 1 neu co loi."""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doi_thu_sfda as S
import global_dnn as GD

torch.manual_seed(20261107); rng = np.random.default_rng(20261107)
torch.set_default_dtype(torch.float64)
loi = []


def kiem(ten, dk, tt=""):
    print(("DAT  " if dk else "LOI  ") + ten, tt)
    if not dk:
        loi.append(ten)


# ---------- K1
def sil_naive(D, lab):
    L = len(lab); D = D / np.linalg.norm(D, axis=1, keepdims=True); dist = 1 - D @ D.T; s = np.zeros(L)
    for i in range(L):
        own = [j for j in range(L) if lab[j] == lab[i] and j != i]
        if not own:
            s[i] = 1.0; continue
        a = np.mean([dist[i, j] for j in own])
        b = min(np.mean([dist[i, j] for j in range(L) if lab[j] == c]) for c in set(lab) if c != lab[i])
        s[i] = (b - a) / max(a, b)
    return s


D = rng.standard_normal((30, 25)); D -= D.mean(1, keepdims=True)
lab = rng.integers(0, 5, 30); lab[0] = 9                                         # cum mot phan tu
s1 = S.sil_cosine(torch.tensor(D), torch.tensor(lab)).numpy()
kiem("K1 silhouette cosine", np.abs(s1 - sil_naive(D, list(lab))).max() < 1e-10 and s1[0] == 1.0)
kiem("K1 mot cum -> NaN nhu MATLAB", bool(torch.isnan(S.sil_cosine(torch.tensor(D), torch.zeros(30, dtype=torch.long))).all()))

# ---------- K2
C, ns, NCHk, L, NL = 4, 40, 3, 24, 6
model = GD.GlobalDNN(C, ns, nclass=NL, nch=NCHk).double()
for p_ in model.parameters():
    p_.requires_grad_(False)
with torch.no_grad():
    model.l1.weight.copy_(torch.tensor([1.0, 0.7, 0.4]).view(1, 3, 1, 1)); model.l2.weight.normal_(0, 1)
X = torch.tensor(rng.standard_normal((L, 3, C, ns)))
yl = torch.tensor(rng.integers(0, NL, L))
comb, score, val, ind = S.silhouette(model, X, yl)
sub = (model.l1.weight.view(1, 3, 1, 1) * X).sum(1).numpy(); W = model.l2.weight[:, 0, :, 0].detach().numpy()
tots = []
for j in range(NCHk):
    P = np.einsum("c,lcn->ln", W[j], sub); P = P - P.mean(1, keepdims=True); tots.append(sil_naive(P, list(yl.numpy())).sum())
kiem("K2 chon to hop kenh theo tong silhouette", abs(max(tots) - score) < 1e-9 and np.allclose(W[int(np.argmax(tots))], comb.detach().numpy()))


def lg_naive(P, indn, lam, delta):
    """Dich tung dong neighbors_calculation.m (dong 40-117), kem sua (v)."""
    L = P.shape[0]; R = np.corrcoef(P); out = {}
    for i in range(L):
        idx = np.argsort(-R[i], kind="stable"); sc = R[i][idx]
        k = 1
        for c in range(L - 2):                                                   # sua (v): chi L - 2 cot
            if (sc[c + 1] - sc[c + 2]) / sc[c + 1] < delta:
                k += 1
            else:
                break
        nn = [int(j) for j in idx[1:k + 1] if indn[j]]
        if not nn and lam != 0 and indn[i]:
            nn = [int(idx[0])]
        if nn:
            w = ((1 - lam) if (indn[i] or lam == 1) else 1.0) / len(nn)
            out[i] = (nn, w)
    return out


for lam in (0.0, 0.6, 1.0):
    sc_, ind_, (pi, pj, pw) = S.lang_gieng(model, X, yl, lam)
    P = np.einsum("c,lcn->ln", comb.detach().numpy(), sub)
    ref = lg_naive(P, ind_.numpy(), lam, S.DELTA)
    got = {}
    for i, j, w in zip(pi.tolist(), pj.tolist(), pw.tolist()):
        got.setdefault(i, ([], w))[0].append(j)
    ok = set(ref) == set(got) and all(ref[i][0] == got[i][0] and abs(ref[i][1] - got[i][1]) < 1e-12 for i in ref)
    kiem(f"K2 lang gieng, trong so (lam = {lam})", ok, f"so mau co lang gieng {len(got)}/{L}")
# phan vi du khong co do sut: 40 tin hieu cos lech pha 0,001 rad
t = np.linspace(0, 2 * np.pi, ns, endpoint=False)
Xc = np.zeros((40, 3, C, ns));
for i in range(40):
    Xc[i, :, :, :] = np.cos(t + 0.001 * i)
m2 = GD.GlobalDNN(C, ns, nclass=NL, nch=NCHk).double()
for p_ in m2.parameters():
    p_.requires_grad_(False)
with torch.no_grad():
    m2.l1.weight.fill_(1.0); m2.l2.weight.fill_(1.0)
lab40 = torch.tensor(np.arange(40) % 2)                                        # hai cum; thay silhouette bang ham cho moi mau qua mat na
_sil_goc = S.silhouette
S.silhouette = lambda mo, Xx, lb: (mo.l2.weight[0, 0, :, 0], 1.0, torch.ones(Xx.shape[0]), torch.ones(Xx.shape[0], dtype=torch.bool))
try:
    _, ind40, (pi, pj, pw) = S.lang_gieng(m2, torch.tensor(Xc), lab40, 0.6)
    dem = np.bincount(pi.numpy(), minlength=40)
    kiem("K2 khong co do sut: khong loi chi so, k_i = L - 1", bool(ind40.all()) and bool((dem == 39).all()), f"so lang gieng moi mau: nho nhat {dem.min()}, lon nhat {dem.max()}")
except Exception as e:
    kiem("K2 khong co do sut", False, repr(e))
S.silhouette = _sil_goc
_, ind1, cap1c = S.lang_gieng(model, X, torch.zeros(L, dtype=torch.long), 0.6)
kiem("K2 mot cum: diem NaN, mat na toan sai, khong co cap", (not bool(ind1.any())) and len(cap1c[0]) == 0 and math.isnan(S.silhouette(model, X, torch.zeros(L, dtype=torch.long))[1]))

# ---------- K3
lg = torch.zeros(4, 2, requires_grad=True)                                       # xac suat 1/2
T = torch.tensor([0, 1, 0, 1]); ind = torch.tensor([True, True, False, False])
cap0 = (torch.tensor([], dtype=torch.long),) * 2 + (torch.tensor([]),)
tot, sl, ll = S.mat_mat(lg, T, ind, cap0, 0.6, 4)
kiem("K3 so hang tu thich nghi: lam * sl = 0,415888", abs(float(tot) - 0.6 * math.log(2)) < 1e-12 and abs(float(tot) - 0.415888) < 1e-6)
cap1 = (torch.tensor([0, 0, 2]), torch.tensor([1, 3, 0]), torch.tensor([0.2, 0.2, 1.0]))
lg2 = torch.tensor(rng.standard_normal((4, 3)), requires_grad=True); T2 = torch.tensor([0, 2, 1, 1])
lp = torch.log_softmax(lg2, 1).detach().numpy()
ll_tay = -(0.2 * lp[0, 2] + 0.2 * lp[0, 1] + 1.0 * lp[2, 0])
sl_tay = -(lp[0, 0] + lp[1, 2]) / 2
for ms in (4, 4 + 40):
    tot, sl, ll = S.mat_mat(lg2, T2, ind, cap1, 0.6, ms)
    kiem(f"K3 hai thanh phan va tong (mau so {ms})", abs(float(sl) - sl_tay) < 1e-12 and abs(float(ll) - ll_tay / ms) < 1e-12 and abs(float(tot) - (0.6 * sl_tay + ll_tay / ms)) < 1e-12)
tot, _, _ = S.mat_mat(lg2, T2, ind, cap1, 0.6, 4); g, = torch.autograd.grad(tot, lg2)
gs = torch.zeros_like(lg2); eps = 1e-6
for i in range(4):
    for j in range(3):
        d = torch.zeros_like(lg2); d[i, j] = eps
        gs[i, j] = (S.mat_mat(lg2.detach() + d, T2, ind, cap1, 0.6, 4)[0] - S.mat_mat(lg2.detach() - d, T2, ind, cap1, 0.6, 4)[0]) / (2 * eps)
kiem("K3 gradient tong so voi sai phan huu han", float((g - gs).abs().max()) < 1e-8)


def sai_phan(ham, x0, eps=1e-6):
    g_ = torch.zeros_like(x0)
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            d = torch.zeros_like(x0); d[i, j] = eps
            g_[i, j] = (ham(x0 + d) - ham(x0 - d)) / (2 * eps)
    return g_


x0 = lg2.detach(); sai_g = []
for ten_tp, idx_tp in (("sl", 1), ("ll", 2), ("tong", 0)):
    for ten_mn, mn in (("0 < N+ < N", ind), ("tap loc rong", torch.zeros(4, dtype=torch.bool))):
        for ms in (4, 44):
            xg = x0.clone().requires_grad_(True)
            ga, = torch.autograd.grad(S.mat_mat(xg, T2, mn, cap1, 0.6, ms)[idx_tp], xg, allow_unused=True)
            ga = torch.zeros_like(x0) if ga is None else ga
            gb = sai_phan(lambda z: S.mat_mat(z, T2, mn, cap1, 0.6, ms)[idx_tp], x0)
            if float((ga - gb).abs().max()) >= 1e-8:
                sai_g.append((ten_tp, ten_mn, ms))
kiem("K3 gradient rieng sl, ll va tong (2 mat na x 2 mau so = 12 ca)", not sai_g, str(sai_g))
tot, sl, _ = S.mat_mat(lg2, T2, torch.zeros(4, dtype=torch.bool), cap1, 0.6, 4)
kiem("K3 tap loc rong: sl = 0, tong huu han", float(sl) == 0.0 and math.isfinite(float(tot)))


# ---------- K4: luong dieu khien voi cac ham duoc thay bang kich ban
class Gia:
    def load_state_dict(self, sd):
        self.sd = sd


def kich_ban(diem_sau_hoc, n_lop_trong_nhan):
    goc = (S.hoc_vong, S.silhouette, S.lang_gieng, S.du_doan)
    dem = {"hoc": 0}
    def hv(model, sd, X, T, ind, cap, lam, mau_so, p2):                       # nhu ham that: de lai mang da hoc trong model
        dem["hoc"] += 1; moi = {"id": dem["hoc"], "tu": sd}; model.load_state_dict(moi); return moi
    S.hoc_vong = hv
    S.du_doan = lambda model, X: torch.arange(X.shape[0]) % n_lop_trong_nhan
    trang = {"i": -1}

    def sil(model, X, lab):
        sd = getattr(model, "sd", None)
        if isinstance(sd, dict) and "id" in sd:
            return None, diem_sau_hoc[sd["id"] - 1], None, None
        return None, 0.0, None, None
    S.silhouette = sil
    S.lang_gieng = lambda model, X, lab, lam, delta=S.DELTA: (0.0, torch.ones(X.shape[0], dtype=torch.bool), cap0)
    try:
        return S.sfda(Gia(), {"goc": True}, torch.zeros(40, 1), torch.zeros(40, dtype=torch.long), 0.6, 0.6, 40)
    finally:
        S.hoc_vong, S.silhouette, S.lang_gieng, S.du_doan = goc


r = kich_ban([-1, 1, 0.5, 0.5, 0.5], 10)                                         # tu choi, nhan, tu choi x 3; nhan thieu lop -> co mau gia
kiem("K4 mau so [80, 80, 40, 40, 40], co mang, vong 3 hoc tiep tu mang da nhan", r["mau_so"] == [80, 80, 40, 40, 40] and r["sd"] is not None and r["sd"]["id"] == 2 and r["sd"]["tu"] == {"goc": True}, str(r["mau_so"]))
r = kich_ban([-1, -1, -1], 10)
kiem("K4 ba lan tu choi: mau so [80, 80, 80], khong tra mang", r["mau_so"] == [80, 80, 80] and r["sd"] is None, str(r["mau_so"]))
r = kich_ban([-1, -1, -1], 40)
kiem("K4 du lop: khong mau gia, mau so [40, 40, 40]", r["mau_so"] == [40, 40, 40] and not r["gia"])

# ---------- K4b: FBCCA thang ban dau, ca ba lan hoc bi tu choi
goc_ = (S.hoc_vong, S.silhouette, S.lang_gieng, S.du_doan)
fb_ = torch.arange(40) % 7


def _hv(model, sd, X, T, ind, cap, lam, mau_so, p2):
    moi = {"id": 1}; model.load_state_dict(moi); return moi


S.hoc_vong = _hv
S.du_doan = lambda model, X: torch.arange(X.shape[0]) % 5
S.silhouette = lambda model, X, lab: (None, (-1.0 if "id" in getattr(model, "sd", {}) else (5.0 if torch.equal(lab, fb_) else 1.0)), None, None)
S.lang_gieng = lambda model, X, lab, lam, delta=S.DELTA: (S.silhouette(model, X, lab)[1], torch.ones(X.shape[0], dtype=torch.bool), cap0)
try:
    r = S.sfda(Gia(), {"goc": True}, torch.zeros(40, 1), fb_, 0.6, 0.6, 40)
finally:
    S.hoc_vong, S.silhouette, S.lang_gieng, S.du_doan = goc_
kiem("K4b FBCCA thang ban dau, ba vong tu choi: nhanh fbcca, nhan = FBCCA, khong tra mang", r["nhanh"] == "fbcca" and r["sd"] is None and torch.equal(r["Y"], fb_) and r["so_lan"] == 3 and r["score"] == 5.0)

# ---------- K4c: ban truc tuyen, quy tac du phong (viii): moc 1 khong mang -> W0; moc 2 co mang A; moc 3 khong mang -> van A
torch.set_default_dtype(torch.float32)
torch.manual_seed(3)
mt = GD.GlobalDNN(C, ns, nclass=NL, nch=NCHk)
for p_ in mt.parameters():
    p_.requires_grad_(False)
Xt = torch.randn(100, 3, C, ns); sdW0 = {k: v.clone() for k, v in mt.state_dict().items()}
sdA = {k: v + 0.5 * torch.randn_like(v) for k, v in sdW0.items()}


def _dd(sd):
    mt.load_state_dict(sd); return S.du_doan(mt, Xt).numpy()


dW0, dA = _dd(sdW0), _dd(sdA)
kich = iter([None, sdA, None])
goc_chon = S.sfda_chon
S.sfda_chon = lambda model, sd0, X, fb, p2, n_lop: (lambda sd: ({"sd": sd}, {"lam": 0.0}))(next(kich))
try:
    dec_t, moc_t = S.truc_tuyen(mt, sdW0, Xt, torch.zeros(100, dtype=torch.long), 0.6, NL, dW0.copy())
finally:
    S.sfda_chon = goc_chon
mong = np.concatenate([dW0[:60], dA[60:100]])
kiem("K4c du phong truc tuyen: luot 41-60 bang W0, 61-100 bang mang A; moc 40, 60, 80", np.array_equal(dec_t, mong) and [m["a"] for m in moc_t] == [40, 60, 80]
     and [m["du_phong"] for m in moc_t] == [True, False, True] and not np.array_equal(dW0[60:], dA[60:]))
torch.set_default_dtype(torch.float64)

# ---------- K5: chay that mot vong SFDA nho (khong kich ban) de chac ma chay thong va tat dinh
torch.set_default_dtype(torch.float32)
torch.manual_seed(1)
m3 = GD.GlobalDNN(C, ns, nclass=NL, nch=NCHk)
X3 = torch.randn(30, 3, C, ns); fb3 = torch.randint(0, NL, (30,))
sd0 = {k: v.clone() for k, v in m3.state_dict().items()}
for p in m3.parameters():
    p.requires_grad_(False)
torch.manual_seed(2); k1, t1 = S.sfda_chon(m3, sd0, X3, fb3, 0.6, NL)
torch.manual_seed(2); k2, t2 = S.sfda_chon(m3, sd0, X3, fb3, 0.6, NL)
kiem("K5 chay thong va tat dinh voi cung seed", t1 == t2 and bool((k1["Y"] == k2["Y"]).all()), f"lam {t1['lam']}, so lan hoc {t1['so_lan']}, co mang {t1['co_mang']}")
print("TONG:", "DAT" if not loi else f"{len(loi)} LOI: {loi}")
sys.exit(1 if loi else 0)
