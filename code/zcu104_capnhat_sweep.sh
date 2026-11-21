#!/bin/bash
# Sweep do cap nhat B1 moi tren ZCU104: 2 cua so x 2 so luong x 5 tien trinh doc lap, nghi 10 s giua tien trinh.
cd ~/ssvep_zcu104_pkg
source venv/bin/activate
OUT=capnhat_b1moi.jsonl
: > $OUT
uptime
for r in 1 2 3 4 5; do
  for T in 0.6 1.0; do
    for L in 1 4; do
      python zcu104_capnhat.py --T $T --luong $L --seed $((20261107 + r)) >> $OUT || echo "LOI T=$T L=$L r=$r"
      sleep 10
    done
  done
done
uptime
echo SWEEP_DONE
