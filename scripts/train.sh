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

mkdir "logs"
mkdir ${CKPT_DIR}

which python

if [ "$1" == "train" ]; then
    python main.py --name ${NAME} \
                   --root ${ROOT} \
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
                   --max-iters ${MAX_ITERS} \
                   --val-interval ${VAL_INTERVAL} \
                   --optimizer ${OPTIMIZER} \
                   --lr-policy ${LR_POLICY} \
                   --loss-fn ${LOSS_FN} \
                   --ckpt-directory ${CKPT_DIR} \
                   --ckpt "$3" \
                   --resume \
                   >>"logs/${NAME}.log" 2>&1
else
    echo "Invalid argument $1"
fi