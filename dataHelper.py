from __future__ import division
import numpy as np
import os
import pandas as pd
import argparse
import sys
import random
import numbers
import png
from matplotlib.pyplot import imsave, imread
import matplotlib
from PIL import Image
import cv2
matplotlib.use("Agg")
import torchvision.datasets as datasets
from skimage.transform import resize
import ast
import pickle
import csv
import pydicom as dcm
import Augmentor
from tqdm import tqdm
import pathlib
import torch
from torch import randint, manual_seed
from torch.nn import functional as F
from torchvision.transforms.functional import crop
from copy import copy
from pathlib import Path
from collections import defaultdict

import re
from ast import literal_eval as make_tuple
from collections import namedtuple

from collections.abc import Sequence


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if extensions is None or len(extensions) == 0 or any((fname.endswith(ext) for ext in extensions)):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def random_flip(input, axis, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            axis += 1
        return np.flip(input, axis=axis)
    else:
        return input


def random_crop(input, with_fa=False):
    
    ran = random.random()
    if ran > 0.2:
        # find a random place to be the left upper corner of the crop
        if with_fa:
            rx = int(random.random() * input.shape[1] // 10)
            ry = int(random.random() * input.shape[2] // 10)
            return input[:, rx: rx + int(input.shape[1] * 9 // 10), ry: ry + int(input.shape[2] * 9 // 10)]
        else:
            rx = int(random.random() * input.shape[0] // 10)
            ry = int(random.random() * input.shape[1] // 10)
            return input[rx: rx + int(input.shape[0] * 9 // 10), ry: ry + int(input.shape[1] * 9 // 10)]
    else:
        return input


def random_rotate_90(input, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            return np.rot90(input, axes=(1,2))
        return np.rot90(input)
    else:
        return input


def random_rotation(x, chance, with_fa=False):
    ran = random.random()
    if with_fa:
        img = Image.fromarray(x[0])
        mask = Image.fromarray(x[1])
        if ran > 1 - chance:
            # create black edges
            angle = np.random.randint(-50, 50)
            img = img.rotate(angle=angle, expand=1)
            mask = mask.rotate(angle=angle, expand=1, fillcolor=1)
            return np.stack([np.asarray(img), np.asarray(mask)])
        else:
            return np.stack([np.asarray(img), np.asarray(mask)])
    img = Image.fromarray(x)
    if ran > 1 - chance:
        # create black edges
        angle = np.random.randint(-50, 50)
        img = img.rotate(angle=angle, expand=1)
        return np.asarray(img)
    else:
        return np.asarray(img)

"""
simple data class for describing an ROI
"""
roi = namedtuple('roi', ['top', 'bottom', 'dist'])

def parse_roi(roi_string):
    """
    Recover an roi namedtuple from its serialized form.
    """

    # Use regex to extract values
    values = re.findall(r'\((\d+), (\d+)\)', roi_string)
    
    # Convert the extracted values to integers and create the namedtuple
    this_roi = roi(top=(int(values[0][0]), int(values[0][1])), 
               bottom=(int(values[1][0]), int(values[1][1])), 
               dist=(int(values[2][0]), int(values[2][1])))
    
    return this_roi

def do_resize_of_context(sample, target_context, roi_coords, ctx_shape):

        ctx_orig = 100 # technically we could figure this out, but much easier just to use the hardcoded value
        # original_target_context_dim = original_target_context*2
        left_right_ctx = ctx_shape[1] - roi_coords.dist[0]
        top_bottom_ctx = ctx_shape[0] - roi_coords.dist[1]

        if roi_coords.top[0] - ctx_orig < 0:
            right_ctx = ctx_orig
            left_ctx = left_right_ctx - right_ctx
        elif left_right_ctx < ctx_orig * 2:
            left_ctx = ctx_orig
            right_ctx = left_right_ctx - left_ctx
        else:
            left_ctx = right_ctx = ctx_orig

        left_ctx_tgt = min(left_ctx, target_context)
        right_ctx_tgt =  min(right_ctx, target_context)

        if roi_coords.top[1] - ctx_orig < 0:
            bottom_ctx = ctx_orig
            top_ctx = top_bottom_ctx - bottom_ctx
        elif top_bottom_ctx < ctx_orig * 2:
            top_ctx = ctx_orig
            bottom_ctx = top_bottom_ctx - top_ctx
        else:
            top_ctx = bottom_ctx = ctx_orig

        top_ctx_tgt = min(top_ctx, target_context)
        bottom_ctx_tgt = min(bottom_ctx, target_context)

        top = 0 + top_ctx - top_ctx_tgt
        left = 0 + left_ctx - left_ctx_tgt
        height = roi_coords.dist[1] + top_ctx_tgt + bottom_ctx_tgt
        width = roi_coords.dist[0] + left_ctx_tgt + right_ctx_tgt

        if top+height > sample.shape[1] or left+width > sample.shape[2]:
            raise ValueError(f"Crop dimensions are larger than array dimensions. {top},{height}-{sample.shape[1]} and {left},{width}-{sample.shape[2]}")

        cropped_array = sample[:, top:top+height, left:left+width]
        
        return cropped_array

class DatasetFolder(datasets.DatasetFolder):
    def __init__(self, root, loader, augmentation=False, extensions=None, transform=None, classmap=None,
                 target_transform=None, is_valid_file=None, target_size=(224, 224), target_context_size=None):

        super(DatasetFolder, self).__init__(root, loader, extensions,
                                            transform=transform,
                                            target_transform=target_transform, )

        if classmap is None:
            classes, class_to_idx = self._find_classes(self.root)
        else:
            classes = list(classmap.keys())
            class_to_idx = classmap
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))
        
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.augment = augmentation
        self.target_size = target_size
        self.target_context_size = target_context_size

        if self.target_context_size is not None:
            try:
                self.index_df = pd.read_csv(Path(root) / 'index.csv')
                self.index_df['roi_coords_parsed'] = self.index_df['roi_coords'].apply(parse_roi)
                self.index_df['roi_ctx_shape_parsed'] = self.index_df['roi_ctx_shape'].apply(make_tuple)
                # self.index_df['original_img_size'] = self.index_df['png_path'].apply(lambda path: np.array(PIL.Image.open(path)).shape)
            except Exception as e:
                print('[ERROR]: unable to load index.csv with metadata for context window resizing')
                raise e

        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        patient_id = path.split("/")[-1][:-4]
        sample = self.loader(path, allow_pickle=True)

        if self.target_context_size is not None:
            sample = self.__resize_context(sample, patient_id)

        if len(sample.shape) == 3:
            if self.target_size:
                sample = np.stack([resize(sample[0], self.target_size), resize(sample[1], self.target_size)])

            # black out middle block
            # sample[0][150:200, 150:200] = 1

            temp = [sample[0], sample[0], sample[0], sample[1]]
        else:
            if self.target_size:
                sample = resize(sample, self.target_size)
            #  normalize to 0 to 1
            # sample = (sample - np.amin(np.abs(sample))) / (np.amax(np.abs(sample))-np.amin(np.abs(sample)))
            #  imagenet normalization
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # temp = []
            # for i in range(3):
            #     temp += [sample-mean[i] / std[i]]

            # print("before transform", sample.shape)
            if self.augment:
                sample = random_rotation(sample, 0.8)
            temp = [sample, sample, sample]
        n = np.stack(temp)
        if self.transform is not None:
            sample = self.transform(n)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample.float(), target, patient_id

    def __resize_context(self, sample, patient_id):

        # How to get an original path from one of the resulting images

        query = self.index_df['path'].str.split('/').str[-1] == patient_id + '.npy'

        record = self.index_df[query]

        assert record.shape[0] == 1
        record = record.loc[record.index[0]]
        # TODO - ['roi_coords'], roi_ctx_shape
        roi_coords = record['roi_coords_parsed']
        context_window_coords = record['roi_ctx_shape_parsed'][1:]

        return do_resize_of_context(sample, self.target_context_size, roi_coords, context_window_coords)


class DatasetFolder_WithReplacement(datasets.DatasetFolder):
    def __init__(self, root, loader, augmentation=False, extensions=None, transform=None, classmap=None,
                 target_transform=None, is_valid_file=None, target_size=(224, 224)):

        super(DatasetFolder_WithReplacement, self).__init__(root, loader, ("npy",),
                                                transform=transform,
                                                target_transform=target_transform, )
        if classmap is None:
            classes, class_to_idx = self._find_classes(self.root)
        else:
            classes = list(classmap.keys())
            class_to_idx = classmap
        
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.augment = augmentation
        self.target_size = target_size
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        index = randint(0, len(self.samples), (1,))[0] #this is the bootstrapping part
        path, target = self.samples[index]
        patient_id = path.split("/")[-1][:-4]
        sample = self.loader(path, allow_pickle=True)
        if len(sample.shape) == 3:
            if self.target_size:
                sample = np.stack([resize(sample[0], self.target_size), resize(sample[1], self.target_size)])

            # black out middle block
            # sample[0][150:200, 150:200] = 1

            temp = [sample[0], sample[0], sample[0], sample[1]]
        else:
            if self.target_size:
                sample = resize(sample, self.target_size)
            #  normalize to 0 to 1
            # sample = (sample - np.amin(np.abs(sample))) / (np.amax(np.abs(sample))-np.amin(np.abs(sample)))
            #  imagenet normalization
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # temp = []
            # for i in range(3):
            #     temp += [sample-mean[i] / std[i]]

            # print("before transform", sample.shape)
            if self.augment:
                sample = random_rotation(sample, 0.7)
            temp = [sample, sample, sample]
        n = np.stack(temp)
        if self.transform is not None:
            sample = self.transform(n)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print("after transform", sample.shape)
        return sample.float(), target, patient_id