# source free
bash dist_train.sh almma/scripts/src_free/srcfree_ripu-mps_linear_acdc.py 8
# test exploration schedule (should be better than `r5_ripu_baseline_strong-int`)
bash dist_train.sh almma/scripts/mps/r20_ripu+mps_linear.py 8
sleep 3