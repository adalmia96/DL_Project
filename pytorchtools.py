#https://raw.githubusercontent.com/Bjarten/early-stopping-pytorch/master/pytorchtools.py

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, gname="generator.pt", dname="discriminator.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            gname (string): Generator name
                            Default: generator.pt
            delta (string): Discriminator name
                            Default: discriminator.pt
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.gen_name = "output/es_" + gname
        self.disc_name = "output/es_" + dname

    def __call__(self, val_loss, generator, discriminator):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, generator, discriminator)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, generator, discriminator)
            self.counter = 0

    def save_checkpoint(self, val_loss, generator, discriminator):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(generator, self.gen_name)
        torch.save(discriminator, self.disc_name)
        self.val_loss_min = val_loss
