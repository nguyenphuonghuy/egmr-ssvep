# EGM-R: unsupervised adaptation of source-model merging coefficients for calibration-free SSVEP decoding

Code and per-subject results for the manuscript "Unsupervised adaptation of source-model merging coefficients
for calibration-free SSVEP decoding on an embedded processor" (D. N. Ton, N. P. Huy; working title; target: Journal of Neural Engineering).

**Status: private working archive.** Comments and file names are still in Vietnamese. Before the repository is made
public at submission, the README will be completed and an archived release will be deposited on Zenodo.

## Names used in the code
| Paper | Code |
|---|---|
| UM (uniform merging, no branch R) | `KTN`, `E0` |
| UM-R (static decoder after the 40-trial warm-up) | `KTNR`, `ktnr` |
| EGM-R (exponentiated-gradient merging, teaching signal from the fused score) | `E4Rb`, branch `b` |
| EGM-R_l (teaching signal from the network logits) | `E4Ra`, branch `a` |
| EGM (coefficient update on the UM base) | `E4` |
| Mechanism analysis on the UM base: 220 l2 parameters / coefficient space, update points every 20 trials (self-generated / fixed labels) | `E1` / `E2` (`E1c` / `E2c`) |
| EGM-R^H, EGM-R^H_z, EGM-R^H_l (entropy-loss controls) | branches `Hs`, `Hz`, `Hl` |
| DH-TT (reproduction of Duan and Hao, 2026) | `dh_tt*` |
| UM-R_c (weights from the mean warm-up confidence) / EGM-R_40 (single fit on the 40 warm-up trials) | `A1` / `A2` |
| Per-trial update of the 220 l2 parameters with a fixed step | `l2` |
| Same, normalized step d_max/(G_t sqrt(T)), Euclidean norms | `J_l2_chuan_hoa` |
| Random coefficients at the EGM-R total-variation distance / reversed update sign | `I1_ngau_nhien` / `I2_nguoc_dau` |
| EGM-R with the session length T replaced by 0.5T, 2T or 1000 | `T_thay` |
| Decision-change rate (DCR) | `P4*` fields |

Internal names (`KTNR`, `E4Rb`, ...) are kept so that result files, logs and stored script hashes stay traceable.

## Layout
- `code/`: decoders, adaptation rules, reproductions of the comparison methods, analysis scripts, MAC counting and ZCU104 timing.
  - `b2e4r.py`: main replay of UM-R, EGM-R and EGM-R_l (this version also covers Dong 2023).
    `run-version-benchmark-beta/b2e4r.py` is the version used for the Benchmark and BETA runs.
  - `b2e4r_entropy*.py`: entropy-loss control (runner, implementation tests, analysis).
  - `b2e4r_hoanvi*.py`: trial-order permutation analysis.
  - `dh_tt*.py`, `doi_thu_*.py`, `rcs_*.py`, `ensemble_dnn*.py`, `cca_net_*.py`: reproductions of the comparison methods.
  - `zcu104_*.py`, `zcu104_*.sh`: timing and memory measurement on the ZCU104 processing system
    (`zcu104_khoi_dong*`: one warm-up trial with 34 or 69 source modules).
  - `b2e4r_dclam.py`: UM-R_c and EGM-R_40 controls. `b2e4r_l2_T.py`: per-trial l2 update, session-length sensitivity and
    shared-network accuracy on the first 40 trials. `b2e4r_hoanvi.py` and `doi_thu_oacca.py` also cover Dong 2023.
  - `phan_tich_bo_sung.py`, `phan_tich_l2_T.py`, `phan_bo_itr.py`, `ve_phan_bo.py`: analyses of these runs, per-subject
    distribution, information transfer rate and the per-subject figure.
  - `b2e4r_huong.py`: direction controls (random coefficients at the same total-variation distance, reversed sign) and the
    per-trial l2 update with a normalized step. `zcu104_mot_lan_l2*`: device cost of the single fit at trial 40 and of the l2
    update. `phan_tich_huong.py`: analysis of these runs.
  - `phan_tich_khoi.py`: differences by block of 40 trials (EGM-R minus UM-R and minus EGM-R_40) and Bonferroni-adjusted
    intervals for the primary comparisons. It also counts the trials corrected and broken per subject and reports the absolute
    information transfer rate. All values are computed from the stored per-trial decisions.
  - Training-seed check of the primary comparison (supplementary section S5). `global_dnn_huanluyen.py`, `ktnr_buoc1.py` and
    `b1moi_lai.py` read the training seed from the environment variable `B2_SEED` (default 20261107, the seed of the paper).
    `b2e4r.py --bo-json-goc` skips the regression check of the UM and EGM results (`E0`, `E4`) against the earlier
    `b1gop` result files, which exist only for seed 20261107. The other checks stay active: source selection, weight hashes,
    the scores of the two branches, trial labels and order, and UM-R against the KTN-R result file. `b2seed_chinh.pbs` retrains the base network, runs UM-R and then EGM-R for one seed. `b2seed_kiem.pbs` and
    `so_kiem_seed.py` checked, before these runs, that the default seed reproduces the stored results (base-network weights of
    BETA fold 1 bit-identical, UM-R and EGM-R decisions of Benchmark folds 1 and 2 identical). `phan_tich_seed.py` computes the
    per-seed and three-seed-mean differences. For seeds 20261108 and 20261109, the Benchmark base networks were trained with
    `global_dnn_huanluyen.py`; the seed-20261107 Benchmark networks came from `global_dnn_thichnghi.py`, which calls the same
    training function.
- `results/`: per-subject JSON results (seed 20261107), analysis outputs and raw ZCU104 timing sessions, in the folder layout expected by
  the analysis scripts (`ket-qua-b2e4r/`, `ket-qua-b1moi/` for UM-R and static baselines, `ket-qua-b1gop/d6/` for OACCA-NQ and SFDA-TT, `ket-qua-b2e4r/bo-sung-20261120/` for the control,
  sensitivity, Dong 2023 permutation and OACCA-NQ runs on Dong 2023, `ket-qua-b2e4r/bo-sung-20261121/` for the direction
  controls and the normalized-step l2 update, `ket-qua-b2e4r/bo-sung-20261122/` for the block analysis, `ket-qua-b2e4r/seed/` for the UM-R and EGM-R runs with training
  seeds 20261108 and 20261109, their job manifests (`hoso_*`: script SHA-256, GPU, package list), the pre-run check (`kiem_19107/`)
  and the seed analysis).
- `SHA256SUMS.txt`: checksums of every file in this archive.

No EEG data, network weights or logits are included.

The code was tidied after the runs. Comments were edited, and the analysis and summary scripts no longer compute the
internal decision labels and bookkeeping fields used during development; these fields were also removed from the result
files. Estimates, confidence intervals and per-subject values are unchanged: rerunning the analysis scripts on the included
result files reproduces the stored analysis outputs. The script SHA-256 recorded inside the result files refers to the
versions before these edits.

### Versions that produced the results
Some scripts were renamed or edited after the runs. The table maps each run version, identified by the SHA-256 recorded in
the result files or in the session manifest, to the file in this archive.

| Run | SHA-256 of the run version | File in this archive |
|---|---|---|
| EGM-R on Benchmark and BETA | `b2e4r.py` eb6f7225ee62093865f13c9cf5caa0ebe4dc00e092f44e3705be69abf6962dad, `b1moi_lai.py` 20f0e571b999d17f0465d1ac5ed7d5bb6d34c2b79da48e085ebf54ac94117fd5 | `code/b2e4r.py`, `code/b1moi_lai.py` |
| EGM-R on Dong 2023 (adds the Dong 2023 loader) | `b2e4r.py` e56f4877ff9e0b36b430e441ce36ce2d37abf9d78d7f46f6bd26d8daffc918b9, `b1moi_lai.py` 7552cd145f201dea434401b5f962664492a401eea792f731269d80f3b4a1dc5f | `code/b2e4r.py`, `code/b1moi_lai.py` |
| ZCU104 single fit at trial 40 and per-trial l2 update | 26cbffb2cfa1e0aacd1d8f5de7d00166900a5e6c40eb522609b353b79dd0493b (run under an earlier file name; the session manifest lists it under the current name) | `code/zcu104_mot_lan_l2.py` |
| Training seeds 20261108 and 20261109 (all three datasets) | listed in `results/ket-qua-b2e4r/seed/hoso_*/sha256.txt`; the Python files are identical to those in this archive, and `b2seed_chinh.pbs` differs only by one edited comment | `code/` (same names) |
| ZCU104 warm-up trial | a73259fd22be3ae3db354c3a2688a078aa01a46e1416096d3184b5c0bacd5e4c | `code/zcu104_khoi_dong.py` |

The byte counter of the per-trial l2 update (`code/zcu104_mot_lan_l2.py`, mode `l2`) adds the l2 weights and biases twice,
once in `tam` and once in `trang_thai`. The stored total of 58 452 bytes therefore includes 880 duplicated bytes. The manuscript
reports 57 572 bytes, that is, `kich_hoat_duy_nhat + tam + trang_thai - 880` from `mot_lan_l2.jsonl`. The byte counts are sums
of the tensors listed by each counter, not the memory of the process.

## Data
Benchmark (Wang et al., 2017), BETA (Liu et al., 2020) and Dong 2023 (Zenodo, CC BY-NC 4.0) are public; see the manuscript for references.

## License
MIT for the authors' code and the result files, except the files listed in `NOTICE`, which are ports of GPL-3.0 code and remain under GPL-3.0.
