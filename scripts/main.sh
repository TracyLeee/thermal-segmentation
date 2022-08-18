#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ..

NAME="$2"
ROOT="data/freiburg"

TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=1
TEST_BATCH_SIZE=1

MAX_ITERS=40000
INFO_INTERVAL=10
VAL_INTERVAL=1000

MANUAL_SEED=0

OPTIMIZER="sgd"
LR_POLICY="step"
LOSS_FN="cross_entropy"

CKPT_DIR="checkpoints"
RESULTS_DIR="seg_results"

mkdir "logs"
mkdir ${CKPT_DIR}

which python

# --manual-seed should be removed when resuming the training
# But the problem is torch._C.Generator object cannot be deepcopied...
# This could be a feature expected to be added later in pytorch
# https://github.com/pytorch/pytorch/issues/43672

if [ "$1" == "train" ]; then
    python main.py --name ${NAME} \
                   --root ${ROOT} \
                   --batch-size ${TRAIN_BATCH_SIZE} \
                   --val-batch-size ${VAL_BATCH_SIZE} \
                   --max-iters ${MAX_ITERS} \
                   --info-interval ${INFO_INTERVAL} \
                   --val-interval ${VAL_INTERVAL} \
                   --manual-seed ${MANUAL_SEED} \
                   --optimizer ${OPTIMIZER} \
                   --lr-policy ${LR_POLICY} \
                   --loss-fn ${LOSS_FN} \
                   --ckpt-directory ${CKPT_DIR} \
                   >"logs/${NAME}.log" 2>&1
elif [ "$1" == "resume" ]; then
    python main.py --name ${NAME} \
                   --root ${ROOT} \
                   --batch-size ${TRAIN_BATCH_SIZE} \
                   --val-batch-size ${VAL_BATCH_SIZE} \
                   --max-iters ${MAX_ITERS} \
                   --info-interval ${INFO_INTERVAL} \
                   --val-interval ${VAL_INTERVAL} \
                   --manual-seed ${MANUAL_SEED} \
                   --optimizer ${OPTIMIZER} \
                   --lr-policy ${LR_POLICY} \
                   --loss-fn ${LOSS_FN} \
                   --ckpt-directory ${CKPT_DIR} \
                   --ckpt "$3" \
                   --resume \
                   >>"logs/${NAME}.log" 2>&1
elif [ "$1" == "test" ]; then
    python main.py --name ${NAME} \
                   --root ${ROOT} \
                   --batch-size ${TEST_BATCH_SIZE} \
                   --loss-fn ${LOSS_FN} \
                   --ckpt-directory ${CKPT_DIR} \
                   --ckpt "$3" \
                   --test \
                   --seg-results-directory ${RESULTS_DIR}\
                   >"logs/${NAME}_test.log" 2>&1      
else
    echo "Invalid argument $1"
fi