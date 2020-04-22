#!/bin/bash

echo "Submitting job to GCP"

DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd $DIR/..

# PROJECT_ID: id of project
PROJECT_ID=$(gcloud config list project --format "value(core.project)")

#BUCKET_ID: id of storage bucket for model
BUCKET_ID=dl-final-project

IMAGE_TAG=latest

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
#IMAGE_REPO_NAME=mnist_pytorch_custom_container
IMAGE_REPO_NAME=pytorch_dl_project

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=gans_training_job_$(date +%Y%m%d_%H%M%S)

# These variables are passed to the docker image
#JOB_DIR=gs://${BUCKET_ID}/models
# Note: these files have already been copied over when the image was built
TRAIN_FILE=news.2009.en.shuffled
WE_FILE=glove.6B.100d.w2v.txt
MODEL=wgantwod

gcloud beta ai-platform jobs submit training ${JOB_NAME} \
	--master-image-uri ${IMAGE_URI} \
	--config config.yaml \
	-- \
	--train-file ${TRAIN_FILE} \
	--we-file ${WE_FILE} \
	--model ${MODEL} \
	--train-epochs=10000 \
	--batch-size=32 \
    --word-vector-length 100 \
    --sequence-length 100 \
    --dimensionality-REDUNDANT 100 \
    --num-data 250000

echo "You may type Ctrl-C if you wish to view the logs online instead."
# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}
