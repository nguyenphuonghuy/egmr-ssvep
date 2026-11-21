#!/bin/bash
# Do tre ban suy luan that kh_deu tren ZCU104 (PS, Cortex-A53): 2 so luong x 7 tien trinh doc lap, nghi 10 s giua tien trinh.
cd ~/ssvep_zcu104_pkg
source venv/bin/activate
rm -f khdeu_dotre_L*_r*.json
uptime
for r in 1 2 3 4 5 6 7; do
  for L in 1 4; do
    OMP_NUM_THREADS=$L OPENBLAS_NUM_THREADS=$L MKL_NUM_THREADS=$L python kh_deu_suy_luan.py --do-tre --lap 200 --out khdeu_dotre_L${L}_r${r}.json > /dev/null || echo "LOI L=$L r=$r"
    sleep 10
  done
done
uptime
echo SWEEP_DONE
