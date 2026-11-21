#!/bin/bash
# Cong R nhanh b1gop tren ZCU104: 7 tien trinh doc lap x (cau hinh quyet dinh: T 1,0 s, 1 luong) + bao kem (1,0 s 4 luong; 0,6 s 1 va 4 luong), nghi 10 s.
cd ~/ssvep_zcu104_pkg
source venv/bin/activate
OUT=b1gop_eg.jsonl
: > $OUT
uptime
for r in 1 2 3 4 5 6 7; do
  for T in 1.0 0.6; do
    for L in 1 4; do
      python zcu104_b1gop_eg.py --T $T --luong $L --seed $((20261107 + r)) >> $OUT || echo "LOI T=$T L=$L r=$r"
      sleep 10
    done
  done
done
uptime
echo SWEEP_DONE
