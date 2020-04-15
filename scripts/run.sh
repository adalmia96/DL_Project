#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd $DIR/..

if [[ $1=='preprocess' ]] || [[ $1=='all' ]]
then
    python main.py --train-file example.train --we-file example_embeddings --mode preprocessing
fi

if [[ $1=='train' ]] || [[ $1=='all' ]]
then
    python main.py --train-file example.train --we-file example_embeddings --model wgantwod --train-epochs 1 \
        --train-d-iters 1 --mode train
fi

if [[ $1=='test' ]] || [[ $1=='all' ]]
then
    python main.py --train-file example.train --we-file example_embeddings --model wgantwod --mode test
fi
