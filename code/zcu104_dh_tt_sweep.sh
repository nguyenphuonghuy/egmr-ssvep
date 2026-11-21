#!/bin/bash
# Sweep do DH-TT tren ZCU104, cung khung voi zcu104_b2e4r_sweep.sh. Dung: bash zcu104_dh_tt_sweep.sh <ten phien>. Tao phien_dhtt_<ten>/, khong ghi de.
# Truoc sweep: mot lan --kiem (n 200) phai dat (ma thoat 0), khong thi dung. Sau do 7 tien trinh doc lap moi cau hinh:
# 1,0 s 1 luong (quyet dinh), 1,0 s 4 luong, 0,6 s 1 luong (bao kem). Cooldown 10 s.
cd ~/ssvep_zcu104_pkg/doi_thu || exit 1
D=phien_dhtt_$1
[ -e $D ] && { echo "da co $D: khong ghi de"; exit 1; }
mkdir $D
source ../venv/bin/activate
sha256sum zcu104_dh_tt.py dh_tt_loi.py zcu104_b1gop_eg.py global_dnn.py b1_diem.py du_lieu_ben/benchmark_S1.npz > $D/sha256_tren_board.txt
uptime > $D/uptime_dau.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python zcu104_dh_tt.py --T 1.0 --luong 1 --n 200 --kiem > $D/kiem.json 2>> $D/loi.log
k=$?; echo -e "kiem\t$k" >> $D/trang_thai.tsv
[ $k -eq 0 ] || { echo "CONG KIEM KHONG DAT ($k): dung"; exit 2; }
sleep 10
for cfg in "1.0 1" "1.0 4" "0.6 1"; do
  set -- $cfg; T=$1; L=$2
  for r in 1 2 3 4 5 6 7; do
    OMP_NUM_THREADS=$L OPENBLAS_NUM_THREADS=$L MKL_NUM_THREADS=$L python zcu104_dh_tt.py --T $T --luong $L --tien-trinh $r --seed $((20261109 + r)) >> $D/dhtt.jsonl 2>> $D/loi.log
    echo -e "T=$T L=$L r=$r\t$?" >> $D/trang_thai.tsv
    sleep 10
  done
done
uptime > $D/uptime_cuoi.txt
echo SWEEP_DONE
