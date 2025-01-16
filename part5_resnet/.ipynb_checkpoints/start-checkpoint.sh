#!/bin/bash

RANDOM_SEED=100
CLAMP_MIN=1
CLAMP_MAX=1
CUDA_GPU="0,1,2,3"
BRIEF="fifth-seed$RANDOM_SEED-clamp$CLAMP_MIN-$CLAMP_MAX"
SCREEN_NAME="conv-$BRIEF-$RANDOM_SEED-$CLAMP_MIN-$CLAMP_MAX"

mkdir -p log

export PYTHONUNBUFFERED=1

echo "Starting screen $SCREEN_NAME"
#-d: Start screen in detached mode, meaning it will run in the background.
#-m: Force the creation of a new screen session, even if one already exists.
#edirects standard error (stderr) to standard output (stdout). This ensures that both error messages and regular output are captured together.
screen -dmS "$SCREEN_NAME" bash -c "python3 cifar10-onelayer-conv-aio.py --brief $BRIEF --random_seed $RANDOM_SEED --clamp_min $CLAMP_MIN --clamp_max $CLAMP_MAX --cuda_gpu $CUDA_GPU 2>&1 | tee log/$SCREEN_NAME-$(date +'%Y%m%d_%H%M%S').txt"
