#!/bin/bash
# Chay thu tren ZCU104 (khong phai so do chinh thuc): kiem tuong duong rut gon va tham do thoi gian 3 luot moi phuong phap de uoc tong gio sweep.
#   nohup bash zcu104_doithu_thu.sh > thu_doithu.log 2>&1 &
PY=~/ssvep_zcu104_pkg/venv/bin/python
cd ~/ssvep_zcu104_pkg/doi_thu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for pp in fbcca rcs ed22; do
  $PY zcu104_doithu_ktnr.py --pp $pp --T 1.0 --luong 4 --kiem --kiem-luot 40 >> thu_doithu.jsonl; echo "kiem $pp ma=$? $(date +%H:%M:%S)"
done
$PY zcu104_doithu_ktnr.py --pp ccanet --T 1.0 --luong 4 --kiem --kiem-luot 5 >> thu_doithu.jsonl; echo "kiem ccanet ma=$? $(date +%H:%M:%S)"
for L in 4 1; do
  export OMP_NUM_THREADS=$L OPENBLAS_NUM_THREADS=$L MKL_NUM_THREADS=$L
  for pp in ktnr fbcca rcs ed22 ccanet edgoc; do
    $PY zcu104_doithu_ktnr.py --pp $pp --T 1.0 --luong $L --n 3 --khoi-dong 1 >> thu_doithu.jsonl; echo "tham do $pp L=$L ma=$? $(date +%H:%M:%S)"
  done
done
echo THU_DONE
