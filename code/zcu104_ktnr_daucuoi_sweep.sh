#!/bin/bash
# Do dau-cuoi KTN-R tren ZCU104: 2 cua so x 2 so luong x 7 tien trinh, nghi 10 s.
cd ~/ssvep_zcu104_pkg/doi_thu
source ../venv/bin/activate
OUT=ktnr_daucuoi.jsonl
: > $OUT
sha256sum zcu104_ktnr_daucuoi.py global_dnn.py b1_diem.py trongso_T10/fold017_trongso.pt
uptime
for r in 1 2 3 4 5 6 7; do
  for T in 1.0 0.6; do
    for L in 1 4; do
      python zcu104_ktnr_daucuoi.py --T $T --luong $L --seed $((20261107 + r)) >> $OUT || echo "LOI T=$T L=$L r=$r"
      sleep 10
    done
  done
done
uptime
echo SWEEP_DONE
