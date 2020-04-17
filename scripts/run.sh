#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd $DIR/..

TRAIN_FILE='news.2009.en.shuffled'
WE_FILE='glove.6B.50d.w2v.txt'
SENTENCE_LENGTH=50
WV_LENGTH=50
MODEL='wgantwod'
DIM_REDUNDANT=50 # Remove after replacing with SENTENCE_LENGTH and WV_LENGTH in wgantwoo

if [[ $1=='preprocess' ]] || [[ $1=='all' ]]
then
    python main.py --train-file $TRAIN_FILE --we-file $WE_FILE --sequence-length $SENTENCE_LENGTH --word-vector-length $WV_LENGTH \
    --model $MODEL \
    --num-data 100 \
    --mode preprocess
fi

if [[ $1=='train' ]] || [[ $1=='all' ]]
then
    python main.py --train-file $TRAIN_FILE --we-file $WE_FILE --sequence-length $SENTENCE_LENGTH --word-vector-length $WV_LENGTH \
    --model $MODEL \
    --batch-size 64 \
    --train-epochs 2 \
    --train-d-iters 5 \
    --train-g-iters 1 \
    --lambda-term 10 \
    --learning-rate 0.0001 \
    --dimensionality-REDUNDANT $DIM_REDUNDANT \
    --mode train
fi

if [[ $1=='test' ]] || [[ $1=='all' ]]
then
    python main.py --train-file $TRAIN_FILE --we-file $WE_FILE --model $MODEL \
    --test-num-images 64 \
    --dimensionality-REDUNDANT $DIM_REDUNDANT \
    --mode test
fi
