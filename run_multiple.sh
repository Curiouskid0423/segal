# RIPU without MTL and with stronger init (to show that MTL should help RIPU sampling)
bash dist_train.sh almma/scripts/mps/r5_ripu_no-mtl_baseline_strong-init.py 8
sleep 3
bash dist_train.sh almma/scripts/mps/r5_ripu_baseline_strong-init.py 8
sleep 3
# test exploration schedule (should be better than `r5_ripu_baseline_strong-int`)
bash dist_train.sh almma/scripts/mps/r5_ripu+mps_contant.py 8
sleep 3
bash dist_train.sh almma/scripts/mps/r5_ripu+mps_linear.py 8
sleep 3
bash dist_train.sh almma/scripts/mps/r10_ripu+mps_linear.py 8
sleep 3