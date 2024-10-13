"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import copy
import pandas as pd
from data import CustomDatasetDataLoader
from data.datahelpers import pil_loader
from data.templates import DatasetFromDataframe
import os
import torch
from sklearn.model_selection import train_test_split
import time
import tqdm
from torch.utils.data.dataloader import default_collate
import rasterio


class BaseDataset(data.Dataset, ABC):
    """Base class for datasets. Each dataset must have a Pandas Dataframe with columns ['qclass, 'class', 'path'] for every item.
    'qclass' is mandatory for training datasets, it will be generated if not present.
    Data loading is performed here, children classes must therefore define dataset files before calling super().__init__
    Method prepare_epoch returns a dataloader for one epoch, depending on the sampling strategy:
        - None: samples a single example at a time, as in the standard __getitem__ method
        - augment: same, with additional data augmentation methods
        - pair: samples two examples from the same class for siamese learning
        - random: samples a query, a positive and nnum negatives samples for contrastive learning (q, p, n1, n2, ...nnum)
        with random sample selection
        - distance: samples a query, a positive and nnum negatives samples for contrastive learning (q, p, n1, n2, ...nnum)
        with desc distance based selection
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.transform = transforms.Compose((
            transforms.Resize(opt.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))
        self.val_transform = transforms.Compose((
            transforms.Resize((opt.imsize, opt.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean,
                                 std=opt.std)
        ))
        self.opt = opt
        self.net = None
        self.name = type(self).__name__.split('dataset')[0]
        self.unique_classes = None
        self.loader = pil_loader

        self.files = []

        # utility attributes used for val and test datasets (exhaustive patch extraction)
        self.current_frame = 0
        self.current_frame_size = [0, 0]
        self.current_patch = [0, 0]
        self.imsize = opt.imsize

        self.load_data()

        print("dataset {} was created, {} samples".format(type(self).__name__, len(self)))

    @abstractmethod
    def load_data(self):
        """ Loads dataset from disk
        """
        return

    def get_frame(self):
        # randomly pick frame
        picked = np.random.randint(len(self.files))
        data = np.load(self.files[picked]).astype(int)
        #img, target = data[:4], data[4]
        #img, target = data[:5], data[5]
        img, target = data[0], data[1]
        valid = target != -1
        return img.astype(np.uint8), target, valid

    def get_patch(self, img, target, valid):
        # Get random patch
        nbands, height, width = img.shape
        cropsize = self.imsize

        potential_rows = np.arange(height - cropsize)
        randrow = np.random.choice(potential_rows)
        randcol = np.random.choice(np.arange(width - cropsize))

        image = img[:, randrow:randrow + cropsize, randcol:randcol + cropsize]
        target = target[randrow:randrow + cropsize, randcol:randcol + cropsize]

        image = torch.from_numpy(image)
        target = torch.from_numpy(target).unsqueeze(0)

        return image, target
