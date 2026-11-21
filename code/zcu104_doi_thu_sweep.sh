#!/bin/bash
# THI NGHIEM B tren ZCU104. Chay trong ~/ssvep_zcu104_pkg/doi_thu (du_lieu_ben/, trongso_T10/ da chep).
# Dung: bash zcu104_doi_thu_sweep.sh <ten phien>   (thu muc ket qua phien_<ten>/ phai chua co: khong ghi de so do cu)
# B1, B2: 7 tien trinh doc lap x (1,0 s 1 luong = cau hinh quyet dinh; 1,0 s 4 luong, 0,6 s 1 luong: bao kem), nghi 10 s.
# B3: vong tong hop a = 40, 220 (5 lan); tau_phu fold 17, 27 x a = 40, 220 (3 lan); moc tron ven fold 17, 27 x a = 40, 220 (tran 30 phut);
#     bo nho: lan chay rieng b3bonho cung seed.
# Moi ca ghi mot dong trang thai vao trang_thai.tsv (phan, cau hinh, ma thoat); stderr vao loi.log. Ca loi/het gio duoc giu, khong bo.
cd ~/ssvep_zcu104_pkg/doi_thu || exit 1
source ../venv/bin/activate
D=phien_${1:?can ten phien}
[ -e "$D" ] && { echo "$D da co: khong ghi de"; exit 1; }
mkdir "$D"
OUT=$D/doi_thu.jsonl; TT=$D/trang_thai.tsv; ERR=$D/loi.log
uptime | tee $D/uptime_dau.txt
sha256sum *.py du_lieu_ben/* trongso_T10/* > $D/sha256_tren_board.txt
chay() {   # chay <nhan> <lenh...>
  local nhan=$1; shift
  "$@" >> $OUT 2>> $ERR; local rc=$?
  printf "%s\t%s\n" "$nhan" "$rc" >> $TT
}
for L in 1 4; do
  export OMP_NUM_THREADS=$L OPENBLAS_NUM_THREADS=$L MKL_NUM_THREADS=$L
  for r in 1 2 3 4 5 6 7; do
    for T in 1.0 0.6; do
      if [ $L = 4 ] && [ $T = 0.6 ]; then continue; fi
      for P in b1 b2; do
        chay "$P T=$T L=$L r=$r" python zcu104_doi_thu.py --phan $P --T $T --luong $L --tien-trinh $r --seed $((20261107 + r))
        sleep 10
      done
    done
  done
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for A in 40 220; do
  chay "b3vong a=$A" python zcu104_doi_thu.py --phan b3vong --a $A --lap 5
  for F in 17 27; do
    chay "b3phu a=$A f=$F" python zcu104_doi_thu.py --phan b3phu --a $A --fold $F --lap 3
  done
done
for F in 17 27; do
  for A in 40 220; do
    chay "b3moc a=$A f=$F" timeout 2100 python zcu104_doi_thu.py --phan b3moc --a $A --fold $F     # 2100 s: chan du phong cua tran 1800 s
    sleep 10
  done
done
for F in 17 27; do
  for A in 40 220; do
    chay "b3bonho a=$A f=$F" timeout 2100 python zcu104_doi_thu.py --phan b3bonho --a $A --fold $F
  done
done
uptime | tee $D/uptime_cuoi.txt
echo SWEEP_DONE
