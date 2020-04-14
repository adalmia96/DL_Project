#!/bin/bash

echo "Submitting job to GCP"

# PROJECT_ID: id of project
PROJECT_ID=black_stream_273918

#BUCKET_ID: id of storage bucket for model
BUCKET_ID=dl-final-project

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=pytorch_dl_project

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=gans_training_job_$(date +%Y%m%d_%H%M%S)

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=us-central1

# These variables are passed to the docker image
JOB_DIR=gs://${BUCKET_ID}/models/base
# Note: these files have already been copied over when the image was built
TRAIN_FILE=fake.train
TEST_FILE=fake.dev

gcloud beta ai-platform jobs submit training ${JOB_NAME} \
	--region ${REGION} \
	--master-image-uri ${IMAGE_URI} \
	--scale-tier BASIC \
	-- \
	--train-file ${TRAIN_FILE} \
	--test-file ${EVAL_FILE} \
	--train-epochs=10 \
	--batch-size=100 \
	--job-dir=${JOB_DIR}

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/model_*
