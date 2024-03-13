import shutil
import os
import torch
import numpy as np
import logging as log

class Checkpoint(object):
    def __init__(self, start_epoch=None, train_loss=None, best_loss=np.inf, state_dict=None, optimizer=None):
        self.start_epoch = start_epoch
        self.train_loss = train_loss
        self.best_loss = best_loss
        self.state_dict = state_dict
        self.optimizer = optimizer

    def save(self, is_best, filename):
        if is_best:
            log.info('Saving the best model at "%s"' % filename)
        else:
            log.info('Saving checkpoint at "%s"' % filename)
        torch.save(self, filename)

    def load(self, filename):
        if os.path.isfile(filename):
            log.info('Resuming checkpoint from "%s"' % filename)
            checkpoint = torch.load(filename, map_location='cpu')

            self.start_epoch = checkpoint.start_epoch
            self.train_loss = checkpoint.train_loss
            self.best_loss = checkpoint.best_loss
            self.state_dict = checkpoint.state_dict
            self.optimizer = checkpoint.optimizer
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)
