#!/bin/bash
# Do H4 tren ZCU104: 3 cau hinh kho (S, N) x 2 cua so x 2 so luong x 7 tien trinh doc lap, nghi 10 s giua tien trinh.
cd ~/ssvep_zcu104_pkg
source venv/bin/activate
OUT=ktn_dotre.jsonl
: > $OUT
uptime
for r in 1 2 3 4 5 6 7; do
  for cfg in "34 40" "69 40" "99 9"; do
    set -- $cfg
    for T in 0.6 1.0; do
      for L in 1 4; do
        python zcu104_ktn_dotre.py --S $1 --N $2 --T $T --luong $L --seed $((20261107 + r)) >> $OUT || echo "LOI S=$1 T=$T L=$L r=$r"
        sleep 10
      done
    done
  done
done
uptime
echo SWEEP_DONE
