"""Hinh phan bo EGM-R - UM-R theo nguoi o 1,0 s (tu results/ket-qua-b2e4r/bo-sung-20261120/phan-bo-itr_20261120.json), ban Anh va ban Viet, kich thuoc in."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = json.load(open(os.path.join(B, "results", "ket-qua-b2e4r", "bo-sung-20261120", "phan-bo-itr_20261120.json")))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9, "axes.linewidth": 0.6,
                     "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "pdf.fonttype": 42})
BO = [("benchmark", "Benchmark", "#2166AC", "o"), ("beta", "BETA", "#B2182B", "s"), ("dong", "Dong 2023", "#1B7837", "^")]
NHAN = {"en": ("EGM-R $-$ UM-R (percentage points)", "median"), "vi": ("EGM-R $-$ UM-R (điểm phần trăm)", "trung vị")}
for ng, (yl, tv) in NHAN.items():
    fig, ax = plt.subplots(figsize=(15 / 2.54, 6 / 2.54))
    rng = np.random.default_rng(20261107)
    for k, (ds, ten, mau, dau) in enumerate(BO):
        v = np.array(d[f"{ds}_10"]["d"])
        x = k + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v, s=10, marker=dau, facecolors="none", edgecolors=mau, linewidths=0.7)
        ax.boxplot(v, positions=[k + 0.38], widths=0.12, showfliers=False, medianprops=dict(color=mau, linewidth=1.2),
                   boxprops=dict(color="black", linewidth=0.6), whiskerprops=dict(color="black", linewidth=0.6), capprops=dict(color="black", linewidth=0.6))
        ax.plot([k - 0.25, k + 0.25], [v.mean()] * 2, color=mau, linestyle="--", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xticks(range(3)); ax.set_xticklabels([b[1] for b in BO]); ax.set_xlim(-0.5, 2.7)
    ax.set_ylabel(yl)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    ra = os.path.join(B, "results", "ket-qua-b2e4r", "bo-sung-20261120", "phan-bo_en.pdf" if ng == "en" else "phan-bo.pdf")
    fig.savefig(ra); fig.savefig(ra.replace(".pdf", ".png"), dpi=200)
    print(ra)
