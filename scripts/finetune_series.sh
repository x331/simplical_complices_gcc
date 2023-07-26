#!/bin/bash
load_path=$1
gpu=$2
ARGS=${@:3}

declare -A epochs=(["usa_airport"]=30 ["h-index"]=30 ["imdb-binary"]=30 ["imdb-multi"]=30 ["collab"]=30 ["rdt-b"]=100 ["rdt-5k"]=100)

for dataset in $ARGS
do
     python train.py --exp FT --model-path saved --tb-path tensorboard --model-ver original --positional-embedding-multi 1 --tb-freq 1  --print-freq 1 --gpu 0 1  --dataset $dataset --finetune --epochs ${epochs[$dataset]} --resume "$load_path/current.pth" --cv
     echo "asodfhasidfhaosidfasdfHelp__________________________-----------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
done