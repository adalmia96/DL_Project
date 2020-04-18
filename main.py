import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import preprocessing as pp

DATA_DIR='./data/'

def get_args():
    """Get arguments from command line."""
    args_parser = argparse.ArgumentParser()

    # Data retrieval arguments
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
    args_parser.add_argument(
        '--num-data',
        help='Number of examples to use from the train file.',
        type=int,
        default=250000)
    # Experiment arguments
    args_parser.add_argument(
        '--model',
        help='Model to experiment with.',
        type=str,
        default='wgantwod')
    args_parser.add_argument(
        '--mode',
        help='Either preprocess, train, or test.',
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
        default=10,
        type=int,
        )
    args_parser.add_argument(
        '--learning-rate',
        help='Learning rate.',
        default=0.0001,
        type=float,
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
        '--dimensionality-REDUNDANT',
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
        from gensim.models.keyedvectors import KeyedVectors
        we_model = KeyedVectors.load_word2vec_format(DATA_DIR+args.we_file, binary=False)

        if args.mode == "preprocess":
            print("in preprocess")
            pp.preprocess(we_model=we_model, training_data=DATA_DIR + args.train_file, num_data=args.num_data, \
                word_array_size=args.sequence_length, word_vector_size=args.word_vector_length)

        if args.mode == "train":
            print("In train")
            models.wgantwod.train(we_model=we_model, batch_size=args.batch_size, epochs=args.train_epochs, \
                d_iters=args.train_d_iters, g_iters=args.train_g_iters, lambda_term=args.lambda_term, \
                lr=args.learning_rate, wv_length=args.word_vector_length, \
                seq_length=args.sequence_length, restore=args.restore, dimensionality=args.dimensionality_REDUNDANT)

        if args.mode == "test":
            models.wgantwod.test(we_model, args.test_num_images, args.dimensionality_REDUNDANT)
    else:
        print("Invalid model!")
    return

if __name__ == "__main__":
    main()
