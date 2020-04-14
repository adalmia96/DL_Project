# Dockerfile

# Install pytorch
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4
# OR
# FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /root

# Installs pandas, and google-cloud-storage.
RUN pip install google-cloud-storage cloudml-hypertune

COPY scripts/docker_entrypoint.sh /root

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["/root/docker_entrypoint.sh"]
