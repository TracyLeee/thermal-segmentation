#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ..

NAME="$2"
ROOT="data/freiburg"
MAX_ITERS=80000
VAL_INTERVAL=1000
OPTIMIZER="sgd"
LR_POLICY="step"
LOSS_FN="cross_entropy"
CKPT_DIR="checkpoints"
RESULTS_DIR="seg_results"
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=1
TEST_BATCH_SIZE=1

mkdir "logs"
mkdir ${CKPT_DIR}

which python

if [ "$1" == "train" ]; then
    python main.py --name ${NAME} \
                   --root ${ROOT} \
                   --batch-size ${TRAIN_BATCH_SIZE} \
                   --val-batch-size ${VAL_BATCH_SIZE} \
                   --max-iters ${MAX_ITERS} \
                   --val-interval ${VAL_INTERVAL} \
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
                   --val-interval ${VAL_INTERVAL} \
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