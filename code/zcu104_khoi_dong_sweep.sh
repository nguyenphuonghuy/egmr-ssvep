#!/bin/bash
# Sweep do pha khoi dong EGM-R tren ZCU104. Dung: bash zcu104_khoi_dong_sweep.sh <ten phien>. Tao phien_khoidong_<ten>/, khong ghi de.
# 7 tien trinh doc lap moi cau hinh: S = 34, 69 nguon; 1,0 s (chinh) va 0,6 s; 1 luong. Cooldown 10 s. Nhu zcu104_b2e4r_sweep.sh.
cd ~/ssvep_zcu104_pkg/doi_thu || exit 1
D=phien_khoidong_$1
[ -e $D ] && { echo "da co $D: khong ghi de"; exit 1; }
mkdir $D
source ../venv/bin/activate
sha256sum zcu104_khoi_dong.py zcu104_b1gop_eg.py global_dnn.py du_lieu_ben/benchmark_S1.npz > $D/sha256_tren_board.txt
uptime > $D/uptime_dau.txt
for cfg in "34 1.0" "69 1.0" "34 0.6" "69 0.6"; do
  set -- $cfg; S=$1; T=$2
  for r in 1 2 3 4 5 6 7; do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python zcu104_khoi_dong.py --S $S --T $T --luong 1 --n 200 --tien-trinh $r --seed $((20261120 + r)) >> $D/khoi_dong.jsonl 2>> $D/loi.log
    echo -e "S=$S T=$T r=$r\t$?" >> $D/trang_thai.tsv
    sleep 10
  done
done
uptime > $D/uptime_cuoi.txt
echo SWEEP_DONE
