import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

def get_args():
    """Get arguments from command line."""
    args_parser = argparse.ArgumentParser()

    # Data files arguments
    args_parser.add_argument(
        '--train-file',
        help='Name of training file. If using Docker/GCS script, must use exact name stored under GCS bucket.',
        #nargs='+',
        required=True)
    args_parser.add_argument(
        '--we-file',
        help='Name of word embeddings file. If using Docker/GCS script, must use exact name stored under GCS bucket.',
        #nargs='+',
        required=True)
     Experiment arguments
    args_parser.add_argument(
        '--model',
        help='Model to experiment with.',
        type=str,
        default='wgan2d')
    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=100)
    args_parser.add_argument(
        '--train-epochs',
        help='Maximum number of training data epochs on which to train.',
        default=5,
        type=int,
        )
    return args_parser.parse_args()

def main():
    """Setup"""
    args = get_args()
    x_train = np.genfromtxt('data/'+args.train_file, delimiter=" ")
    train_loader = DataLoader(dataset=x_train, batch_size=args.batch_size)

    if args.model == 'wgan2d':
        import models.wgantwod
        models.wgan2d.train()
        #models.wgan2d.test()
    elif args.model == 'fake':
        import models.fake
        models.fake.train(train_loader, epochs=args.train_epochs)
        models.fake.test()
    else:
        print("Invalid model!")
    return
