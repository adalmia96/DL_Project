import argparse

def get_args():
    """Get arguments from command line."""
    # Data files arguments
    args_parser.add_argument(
        '--train-file',
        help='Name of training file. If using Docker/GCS script, must use exact name stored under GCS bucket.',
        nargs='+',
        required=True)
    args_parser.add_argument(
        '--test-file',
        help='Name of dev file. If using Docker/GCS script, must use exact name stored under GCS bucket.',
        nargs='+',
        required=True)
    # Experiment arguments
    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=100)
    args_parser.add_argument(
        '--num-epochs',
        help='Maximum number of training data epochs on which to train.',
        default=50,
        type=int,
        )
    return args_parser.parse_args()

def main():
    """Setup"""
    args = get_args()
    print args
    if args.model == 'wgan2d':
        import models.wgantwod
        models.wgan2d.train()
        models.wgan2d.test()
    else if args.model == 'fake':
        import models.fake
        models.fake.train()
        models.fake.test()
    else:
        print("Invalid model!")
    return

if __name__ == "__main__":
    main()
