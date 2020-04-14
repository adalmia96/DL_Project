# GANs for Natural Language Processing

In our final project for deep learning, we experiment with [https://github.com/robert-d-schultz/gan-word-embedding](Robert Schultz's code) under new datasets, GAN models, and word embeddings.

## Setup and Run
Install pytorch and python3. Run the following code for the model. This assumes your data is in `/data` if you're not using the Docker set up.
```
python trainer/main.py --train-dataset example.train --dev-dataset example.dev --train-epochs 5 --model wgan2d --batch-size 10
```

## Docker and GCP
To run docker scripts, ensure you have `docker` and `gsutils` installed. Then run `./scripts/train-cloud.sh`. Ensure your data is in GCP Storage under namespace `/data`.
