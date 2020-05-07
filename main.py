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
        '--generator-file',
        help='Name of generator file output.',
        type=str,
        default='generator.pt')
    args_parser.add_argument(
        '--discriminator-file',
        help='Name of discriminator input or output.',
        type=str,
        default='discriminator.pt')
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
        help='Batch size for each training and evaluation step. Must be divisible by 8.',
        type=int,
        default=64)
    args_parser.add_argument(
        '--train-epochs',
        help='Maximum number of times generator and discriminator iters are run.',
        default=10000,
        type=int,
        )
    args_parser.add_argument(
        '--train-d-iters',
        help='Iterations discriminator is run for within each training epoch.',
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
        '--g-learning-rate',
        help='Generator learning rate.',
        default=0.0001,
        type=float,
        )
    args_parser.add_argument(
        '--d-learning-rate',
        help='Discriminator learning rate.',
        default=0.0001,
        type=float,
        )
    args_parser.add_argument(
        '--word-vector-length',
        help='Dimensionality of a specific word vector. Must be a multiple of 50.',
        default=50,
        type=int,
        )
    args_parser.add_argument(
        '--sequence-length',
        help='Maximum length of a sentence. Must be a multiple of 50.',
        default=50,
        type=int,
        )
    args_parser.add_argument(
        '--test-num-images',
        help='Number of images/sentences to generate when testing the model. Multiple of 32.',
        default=128,
        type=int,
        )
    args_parser.add_argument(
        '--patience',
        help='Early stopping to end training once a plateau is hit. Patience is number \
            of iterations that loss does not improve before model is terminated.',
        default=200,
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

        if args.mode =="image":
            print("generating image")
            pp.create_fancy_image("The quick brown fox jumps over the lazy dog.", we_model=we_model, word_array_size=args.sequence_length, word_vector_size=args.word_vector_length)

        if args.mode == "preprocess":
            print("in preprocess")
            pp.preprocess(we_model=we_model, training_data=DATA_DIR + args.train_file, num_data=args.num_data, \
                word_array_size=args.sequence_length, word_vector_size=args.word_vector_length)

        if args.mode == "train":
            print("In train")
            models.wgantwod.train(we_model=we_model, batch_size=args.batch_size, epochs=args.train_epochs, \
                d_iters=args.train_d_iters, g_iters=args.train_g_iters, lambda_term=args.lambda_term, \
                g_lr=args.g_learning_rate, d_lr=args.d_learning_rate, wv_length=args.word_vector_length, \
                seq_length=args.sequence_length, restore=args.restore, patience=args.patience, \
                discriminator_file=args.discriminator_file, generator_file=args.generator_file)

        if args.mode == "test":
            models.wgantwod.test(we_model=we_model, num_images=args.test_num_images, wv_length=args.word_vector_length, \
            seq_length=args.sequence_length, generator_file=args.generator_file)
    else:
        print("Invalid model!")
    return

if __name__ == "__main__":
    main()
