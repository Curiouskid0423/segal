CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# NOTE: if module cannot be found, export PYTHONPATH to current project dir.
# e.g. `export PYTHONPATH="{$PYTHONPATH}:/home/yutengli/workspace/spring22/al_seg/"`
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_active_learning.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}