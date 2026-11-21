#!/bin/bash
# Sweep do thoi gian dau-cuoi doi thu B1 tren ZCU104.
# Ban 2. Chay trong ~/ssvep_zcu104_pkg/doi_thu:
#   nohup bash zcu104_doithu_ktnr_sweep.sh <ma_phien> > sweep_doithu_<ma_phien>.log 2>&1 &
# Buoc 1: --kiem 40 luot cho ca 6 phuong phap, hai cua so; khac thi dung (ma 5).
# Buoc 2: 6 phuong phap x 2 cua so x 2 luong x 7 tien trinh, nghi 10 s; moi lan do ghi JSON rieng vao <ma_phien>/do/, so ghi <ma_phien>/so_ghi.tsv
# (phien, pp, T, luong, r, seed, n, ma_thoat, tep). Ensemble-DNN ban goc: tham do 3 luot > 1 s thi n = 20; tham do loi thi dung.
PH=${1:?can ma phien}
PY=~/ssvep_zcu104_pkg/venv/bin/python
cd ~/ssvep_zcu104_pkg/doi_thu
[ -e $PH ] && { echo "phien $PH da co: khong ghi de"; exit 2; }
mkdir -p $PH/kiem $PH/do
SG=$PH/so_ghi.tsv
sha256sum zcu104_doithu_ktnr.py zcu104_doithu_ktnr_sweep.sh > $PH/sha.txt
(cd cca_net/goc && sha256sum -c SHA256SUMS.txt) || { echo "MA GOC CCA-NET BI SUA"; exit 3; }
uptime
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for T in 1.0 0.6; do
  for pp in ktnr fbcca rcs ed22 edgoc ccanet; do
    $PY zcu104_doithu_ktnr.py --pp $pp --T $T --luong 4 --kiem --kiem-luot 40 --phien $PH --out $PH/kiem/${pp}_T$T.json
    rc=$?; echo "kiem $pp T=$T ma=$rc $(date +%H:%M:%S)"
    [ $rc -ne 0 ] && { echo "KIEM TUONG DUONG KHONG DAT: $pp T=$T"; exit 5; }
  done
done
echo "KIEM_DAT"
for T in 1.0 0.6; do
  for L in 4 1; do
    export OMP_NUM_THREADS=$L OPENBLAS_NUM_THREADS=$L MKL_NUM_THREADS=$L
    $PY zcu104_doithu_ktnr.py --pp edgoc --T $T --luong $L --n 3 --khoi-dong 1 --phien $PH --out $PH/thamdo_edgoc_T${T}_L$L.json
    [ $? -ne 0 ] && { echo "THAM DO EDGOC LOI T=$T L=$L"; exit 6; }
    tb=$($PY -c "import json; print(json.load(open('$PH/thamdo_edgoc_T${T}_L$L.json'))['tong']['tb_ms'])") || exit 6
    N_ED=200; $PY -c "import sys; sys.exit(0 if float('$tb') > 1000 else 1)" && N_ED=20
    echo "tham do edgoc T=$T L=$L: $tb ms -> n=$N_ED"
    for pp in ktnr fbcca rcs ed22 edgoc ccanet; do
      n=200; [ $pp = edgoc ] && n=$N_ED
      for r in 1 2 3 4 5 6 7; do
        sd=$((20261107 + r)); f=$PH/do/${pp}_T${T}_L${L}_r$r.json
        $PY zcu104_doithu_ktnr.py --pp $pp --T $T --luong $L --n $n --seed $sd --phien $PH --out $f
        rc=$?; printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $PH $pp $T $L $r $sd $n $rc $f >> $SG
        echo "$pp T=$T L=$L r=$r ma=$rc $(date +%H:%M:%S)"
        sleep 10
      done
    done
  done
done
echo SWEEP_DONE
