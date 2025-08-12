"""
Early stopping utility for PyTorch training.

This module provides an early stopping mechanism to prevent overfitting during
RNN training. It monitors validation loss and stops training when the loss
stops improving for a specified number of epochs (patience).

Author: Derek Jinyu Dong (adapted from PyTorch early stopping utilities)
Date: 2024-2025
"""

# early_stopping.py
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def get_state(self):
        """Return a dict representing the current state for serialization."""
        return {
            'patience': self.patience,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_val_loss': float(self.best_val_loss) if self.best_val_loss is not None else None,
            'early_stop': self.early_stop,
            'val_loss_min': float(self.val_loss_min) if self.val_loss_min is not None else None,
            'delta': self.delta,
            'path': self.path,
        }

    def set_state(self, state):
        """Restore state from a dict (as returned by get_state)."""
        self.patience = state.get('patience', self.patience)
        self.verbose = state.get('verbose', self.verbose)
        self.counter = state.get('counter', 0)
        self.best_val_loss = state.get('best_val_loss', None)
        self.early_stop = state.get('early_stop', False)
        self.val_loss_min = state.get('val_loss_min', np.inf)
        self.delta = state.get('delta', 0)
        self.path = state.get('path', 'checkpoint.pt')

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return self.counter

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.counter

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
