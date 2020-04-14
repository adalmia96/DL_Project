#!/bin/bash

cd /root/
git clone https://github.com/adalmia96/DL_Project
cd DL_Project

# Get GCP datasets based on the train and dev files.
numArgs=("$#")
args=("$@")

for (( i=0; i<numArgs; i++ ))
do
    if [[ "${args[i]}" == "--train-file" ]]
    then
        train_file=("${args[i+1]}")
    elif [[ "${args[i]}" == "--test-file" ]]
    then
        test_file=("${args[i+1]}")
    fi
done

if [ -z "$train_file" ] || [ -z "$test_file" ]
then
    echo "Must specify train and test files!"
    exit 1
fi

BUCKET_NAME=dl-final-project

TRAIN_FILE_URI=gs://$BUCKET_NAME/data/$train_file
TEST_FILE_URI=gs://$BUCKET_NAME/data/$test_file
TIMESTAMP=$(date +%s)
MODEL_URI=gs://$BUCKET_NAME/models/model_${TIMESTAMP}

echo "Downloading train and test data"
gsutil cp $TRAIN_FILE_URI ./data/${train_file}
gsutil cp $TEST_FILE_URI ./data/${test_file}

python -u ./main.py ${@}
gsutil cp ./output/model $MODEL_URI
