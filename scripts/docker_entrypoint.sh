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
    elif [[ "${args[i]}" == "--we-file" ]]
    then
        we_file=("${args[i+1]}")
    fi
done

if [ -z "$train_file" ] || [ -z "$we_file" ]
then
    echo "Must specify train and word embedding files!"
    exit 1
fi

BUCKET_NAME=dl-final-project

TRAIN_FILE_URI=gs://$BUCKET_NAME/data/$train_file
WE_FILE_URI=gs://$BUCKET_NAME/data/$we_file
TIMESTAMP=$(date +%s)
MODEL_URI=gs://$BUCKET_NAME/models/iter_${TIMESTAMP}/

echo "Downloading train and word embeddings data"
gsutil cp $TRAIN_FILE_URI ./data/${train_file}
gsutil cp $WE_FILE_URI ./data/${we_file}

# Install packages if needed, should be done
pip install -r requirements.txt
echo -e "import nltk\nnltk.download('punkt')" | python

python ./preprocessing.py
python ./main.py ${@}
gsutil cp ./output/* $MODEL_URI
