#!/bin/bash
gpu=$1
ARGS=${@:2}

compute-sanitizer python train.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu $gpu \
  $ARGS
