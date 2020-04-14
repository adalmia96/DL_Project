# GANs for Natural Language Generation

In our final project for deep learning, we experiment with [Robert Schultz's paper](https://academicworks.cuny.edu/cgi/viewcontent.cgi?article=4093&context=gc_etds) under new datasets, GAN models, and word embeddings.

## Setup and Run
Install pytorch and python3. Run the following code for the model. This assumes your data is in `/data` if you're not using the Docker set up. **Make sure to push your changes to Git before running this script!**
```
python trainer/main.py --train-dataset example.train --dev-dataset example.dev --train-epochs 5 --model wgan2d --batch-size 10
```
Output is saved to the `output` folder.

## Further Setup for GCP
Ensure you have [`gsutils`](https://cloud.google.com/storage/docs/gsutil_install) installed. Put your train and dev data in GCP Storage under namespace `/data`. Then run `./scripts/train-cloud.sh` after adjusting hyperparameters, or submit a hyperparameter job. You can view logs using the script or on the GCP platform. The final model is saved to the `/models` namespace.
## Docker Image
To update the Docker image, ensure you have  `docker` and `gsutils` installed. Then run `./scripts/upload-image.sh`. You should not need to do this unless you need to add extra dependencies, or if you want to avoid installing GPU dependencies if you are running the model on GPU. To avoid uploading to Google Cloud, comment out `docker push`.

### GCP Todos
- Check that everything works with GPU (change from BASIC to GPU instance in `train-cloud.sh`)
- Add another script for hyperparameter tuning (and [support in the code](https://cloud.google.com/ai-platform/training/docs/custom-containers-training#submit_a_hyperparameter_tuning_job))
