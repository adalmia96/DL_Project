# Dockerfile

# Install pytorch
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4
# OR
# FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /root

RUN pip install cloudml-hypertune

COPY scripts/docker_entrypoint.sh /root/docker_entrypoint.sh

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["/root/docker_entrypoint.sh"]
