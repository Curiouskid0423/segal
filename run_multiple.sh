bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr5e-4_no-lr-mult.py 8
sleep 3
bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr5e-4_no-reg.py 8
sleep 3

# bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr1e-2.py 8
# sleep 3
# bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr5e-3.py 8
# sleep 3
# bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr1e-3.py 8
# sleep 3
# bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr1e-3_nowarmup.py 8
# sleep 3
# bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr5e-4.py 8
# sleep 3
# bash dist_train.sh almma/scripts/random_mtl/n2a_no-mtl_baseline_adamw_lr1e-4.py 8
# sleep 3