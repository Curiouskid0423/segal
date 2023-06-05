bash dist_train.sh almma/scripts/n2a_ada_random_baseline_e28.py 8
sleep 3
bash dist_train.sh almma/scripts/n2a_ada_random_mt_lr1e-3_slow-enc.py 8
sleep 3
bash dist_train.sh almma/scripts/n2a_ada_random_mt_lr1e-3_slow-mae.py 8
sleep 3
bash dist_train.sh almma/scripts/n2a_ada_random_mt_lr5e-4_slow-enc.py 8
sleep 3
bash dist_train.sh almma/scripts/n2a_ada_random_mt_lr5e-4_slow-mae.py 8
# bash dist_train.sh almma/scripts/n2a_ada_random_baseline.py 8
# sleep 3
# bash dist_train.sh almma/scripts/n2a_ada_random_multitask.py 8
# sleep 3