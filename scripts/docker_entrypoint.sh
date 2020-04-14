#!/bin/bash

WORKDIR /root
git clone https://github.com/adalmia96/DL_Project
#mv DL_Project /root

# Get GCP datasets based on the train and dev files.
numArgs=("$#")
args=("$@")

for (( i=0; i<numArgs; i++ ))
do
    echo ${args[i]}
    if [[ "${args[i]}" == "--train-file" ]]
    then
        train_file=("${args[i+1]}")
    elif [[ "${args[i]}" == "--test-file" ]]
    then
        test_file=("${args[i+1]}")
    fi
done
echo $train_file

if [ -z "$train_file" ] || [ -z "$test_file" ]
then
    echo "Must specify train and test files!"
    exit 1
fi

BUCKET_NAME=dl-final-project

TRAIN_FILE_URI=gs://${BUCKET_ID}/data/${train_file}
TEST_FILE_URI=gs://${BUCKET_ID}/data/${test_file}

RUN gsutil cp $TRAIN_FILE_URI ./${train_file}
RUN gsutil cp $TEST_FILE_URI ./${test_file}

python -u trainer/main.py $args
