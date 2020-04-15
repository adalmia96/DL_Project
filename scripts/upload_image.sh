#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd $DIR/..

# PROJECT_ID: id of project
PROJECT_ID=$(gcloud config list project --format "value(core.project)")

IMAGE_TAG=latest

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=pytorch_dl_project

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Deploy the docker image to Cloud Container Registry
docker push ${IMAGE_URI}

# Submit your training job
echo "Uploading image"

