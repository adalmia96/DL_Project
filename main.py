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
    # Experiment arguments
    args_parser.add_argument(
        '--model',
        help='Model to experiment with.',
        type=str,
        default='wgantwod')
    args_parser.add_argument(
        '--mode',
        help='Either train or test.',
        type=str,
        default='test')
    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=64)
    args_parser.add_argument(
        '--train-epochs',
        help='Maximum number of times generator and discrimator iters are run.',
        default=10000,
        type=int,
        )
    args_parser.add_argument(
        '--train-d-iters',
        help='Iterations discrimator is run for within each training epoch.',
        default=5,
        type=int,
        )
    args_parser.add_argument(
        '--train-g-iters',
        help='Iterations generator is run for within each training epoch.',
        default=1,
        type=int,
        )
    args_parser.add_argument(
        '--restore',
        help='Option to continue training from loaded model in output.',
        default=False,
        type=bool,
        )
    args_parser.add_argument(
        '--lambda-term',
        help='Gradient penalty hyperparameter.',
        default=1,
        type=int,
        )
    args_parser.add_argument(
        '--lr',
        help='Learning rate.',
        default=1,
        type=int,
        )
    args_parser.add_argument(
        '--word-vector-length',
        help='Dimensionality of a specific word vector.',
        default=50,
        type=int,
        )
    args_parser.add_argument(
        '--sequence-length',
        help='Maximum length of a sentence.',
        default=50,
        type=int,
        )
    args_parser.add_argument(
        '--test-num-images',
        help='Number of images to generate when testing the model.',
        default=64,
        type=int,
        )
    args_parser.add_argument(
        '--dimensionality-REDUDANT',
        help='REDUNDANT, model dimensionality, need to confirm before removing',
        default=50,
        type=int,
        )
    return args_parser.parse_args()

def main():
    """Setup"""
    args = get_args()
    
    print("Running main.py with these parameters: ")
    print(args)

    if args.model == 'wgantwod':
        import models.wgantwod
        if args.mode == "train":
            models.wgantwod.train(batch_size=args.batch_size, epochs=args.train_epochs, d_iters=args.train_d_iters,  \
                g_iters=args.train_g_iters, lambda_term=args.lambda_term, lr=args.lr, wv_length=args.word_vector_length, \
                seq_length=args.sequence_length, restore=args.restore, dimensionality=args.dimensionality_REDUDANT)
        else:
            models.wgantwod.test(args.test_num_images)
    else:
        print("Invalid model!")
    return

if __name__ == "__main__":
    main()
