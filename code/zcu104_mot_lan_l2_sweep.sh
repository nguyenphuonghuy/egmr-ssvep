#!/bin/bash
# Sweep do bo sung 2026-11-21 (K) tren ZCU104. Dung: bash zcu104_mot_lan_l2_sweep.sh <ten phien>. Tao phien_mot-lan-l2_<ten>/, khong ghi de.
# 7 tien trinh doc lap moi cau hinh, 1 luong, 1,0 s: mot_lan voi T = 200 va 120 (20 lan do), l2 voi T = 200 (1000 luot). Cooldown 10 s.
cd ~/ssvep_zcu104_pkg/doi_thu || exit 1
D=phien_mot-lan-l2_$1
[ -e $D ] && { echo "da co $D: khong ghi de"; exit 1; }
mkdir $D
source ../venv/bin/activate
sha256sum zcu104_mot_lan_l2.py zcu104_b2e4r.py zcu104_b1gop_eg.py global_dnn.py du_lieu_ben/benchmark_S1.npz > $D/sha256_tren_board.txt
uptime > $D/uptime_dau.txt
for cfg in "mot_lan 200 20 3" "mot_lan 120 20 3" "l2 200 1000 20"; do
  set -- $cfg; M=$1; TP=$2; N=$3; KD=$4
  for r in 1 2 3 4 5 6 7; do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python zcu104_mot_lan_l2.py --che-do $M --Tphien $TP --n $N --khoi-dong $KD --tien-trinh $r --seed $((20261121 + r)) >> $D/mot_lan_l2.jsonl 2>> $D/loi.log
    echo -e "M=$M T=$TP r=$r\t$?" >> $D/trang_thai.tsv
    sleep 10
  done
done
uptime > $D/uptime_cuoi.txt
echo SWEEP_DONE
