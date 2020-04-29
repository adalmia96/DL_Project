#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd $DIR/..

TRAIN_FILE='news.2009.en.shuffled'
#WE_FILE='glove.6B.100d.w2v.txt'
WE_FILE='jose_100d.w2v.txt'
SENTENCE_LENGTH=50
WV_LENGTH=100
MODEL='wgantwod'

ARGUMENTS="--train-file $TRAIN_FILE --we-file $WE_FILE --sequence-length $SENTENCE_LENGTH --word-vector-length $WV_LENGTH  \
    --model ${MODEL} \
    --num-data 100 \
    --batch-size 64 \
    --train-epochs 2 \
    --train-d-iters 1 \
    --train-g-iters 1 \
    --lambda-term 10 \
    --learning-rate 0.0001 \
    --generator-file generator.pt \
    --discriminator-file discriminator.pt \
    --test-num-images 128 \
    --patience 1
"

if [[ $1 == 'preprocess' ]] || [[ $1 == 'all' ]]
then
    python main.py $ARGUMENTS --mode preprocess
fi

if [[ $1 == 'train' ]] || [[ $1 == 'all' ]]
then
    python main.py $ARGUMENTS --mode train
fi

if [[ $1 == 'test' ]] || [[ $1 == 'all' ]]
then
    python main.py $ARGUMENTS --mode test
fi
