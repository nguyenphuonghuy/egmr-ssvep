#!/usr/bin/env python
"""Do tren ZCU104 (PS Cortex-A53, PyTorch CPU) thoi gian suy luan va mot buoc cap nhat thich nghi cua Global DNN thu nho
(kien truc nhu ma-b1v4/global_dnn.py; 9 dien cuc, 3 dai, 22 kenh dac trung). Trong so ngau nhien: chi do thoi gian.
Muc dich: dinh tran chi phi cap nhat cho B1 moi.
Mot tien trinh = mot (T, so luong); in JSON mot dong. Bien the cap nhat:
  fc   : chi lop FC (khong lan truyen nguoc vao lop truoc)
  l2fc : lop gop kenh (l2) + FC (lan truyen nguoc qua l4, l3)
  all  : toan mang
Lo cap nhat b in {1, 16}. Moi buoc = thuan + nguoc + Adam step (lr 1e-4), che do train (dropout bat).
"""
import argparse, json, resource, time
import torch
import torch.nn as nn

NSB, NT = 3, 40


class GlobalDNN(nn.Module):
    def __init__(self, C, ns, nch, p_cuoi=0.95):
        super().__init__()
        self.l1 = nn.Conv2d(NSB, 1, (1, 1), bias=False)
        self.l2 = nn.Conv2d(1, nch, (C, 1))
        self.d1 = nn.Dropout(0.1)
        self.l3 = nn.Conv2d(nch, nch, (1, 2), stride=(1, 2))
        self.d2 = nn.Dropout(0.1)
        self.l4 = nn.Conv2d(nch, nch, (1, 10), padding="same")
        self.d3 = nn.Dropout(p_cuoi)
        self.fc = nn.Linear(nch * (ns // 2), NT)

    def forward(self, x):
        h = self.l2(self.l1(x))
        h = torch.relu(self.d2(self.l3(self.d1(h))))
        h = self.d3(self.l4(h))
        return self.fc(h.flatten(1))


def mac_giai_tich(C, ns, nch):
    n2 = ns // 2
    f = {"l1": NSB * C * ns, "l2": nch * C * ns, "l3": nch * n2 * nch * 2, "l4": nch * n2 * nch * 10, "fc": nch * n2 * NT}
    thuan = sum(f.values())
    # nguoc: grad trong so moi lop duoc hoc ~ MAC thuan cua lop do; grad dau vao cua lop sau lop hoc dau tien ~ MAC thuan cua lop do
    nguoc = {"fc": f["fc"],
             "l2fc": f["fc"] + 2 * f["l4"] + 2 * f["l3"] + f["l2"] + f["fc"],
             "all": f["fc"] * 2 + 2 * f["l4"] + 2 * f["l3"] + 2 * f["l2"] + f["l1"]}
    return thuan, {k: thuan + v for k, v in nguoc.items()}


def do(fn, lap, khoi):
    for _ in range(khoi):
        fn()
    t = []
    for _ in range(lap):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    t.sort()
    return {"tb_ms": 1e3 * sum(t) / len(t), "trung_vi_ms": 1e3 * t[len(t) // 2], "p95_ms": 1e3 * t[int(0.95 * (len(t) - 1))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--luong", type=int, required=True)
    ap.add_argument("--nch", type=int, default=22)
    ap.add_argument("--lap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    torch.set_num_threads(a.luong); torch.manual_seed(a.seed)
    C, ns = 9, int(round(a.T * 250))
    thuan, capnhat = mac_giai_tich(C, ns, a.nch)
    out = dict(T=a.T, luong=a.luong, nch=a.nch, lap=a.lap, seed=a.seed, torch=torch.__version__, mac_thuan=thuan, mac_capnhat_b1=capnhat)
    m = GlobalDNN(C, ns, a.nch)
    m.eval()
    x1 = torch.randn(1, NSB, C, ns)
    with torch.no_grad():
        out["suy_luan_b1"] = do(lambda: m(x1), a.lap, 20)
    lossf = nn.CrossEntropyLoss()
    for bien in ("fc", "l2fc", "all"):
        for b in (1, 16):
            mm = GlobalDNN(C, ns, a.nch); mm.train()
            for p in mm.parameters():
                p.requires_grad_(False)
            hoc = {"fc": [mm.fc], "l2fc": [mm.l2, mm.fc], "all": [mm]}[bien]
            ps = [p for mod in hoc for p in mod.parameters()]
            for p in ps:
                p.requires_grad_(True)
            opt = torch.optim.Adam(ps, lr=1e-4)
            xb = torch.randn(b, NSB, C, ns); yb = torch.randint(0, NT, (b,))

            def buoc():
                opt.zero_grad(set_to_none=True)
                lossf(mm(xb), yb).backward(); opt.step()
            out[f"capnhat_{bien}_b{b}"] = do(buoc, max(20, a.lap // 2), 5)
            out[f"so_tham_so_hoc_{bien}"] = sum(p.numel() for p in ps)
    out["rss_dinh_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(out))


if __name__ == "__main__":
    main()
