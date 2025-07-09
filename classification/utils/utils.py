# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

# Modified by: Daquan Zhou

'''
- resize_pos_embed: resize position embedding
- load_for_transfer_learning: load pretrained paramters to model in transfer learning
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress.
'''

import os
import sys
import time
import torch
import math

import torch.nn as nn
import torch.nn.init as init
import logging
import os
from collections import OrderedDict
import torch.nn.functional as F
import torch.utils.data

import random
import numpy as np
from typing import Callable
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from functools import partial
from itertools import repeat
from timm.data.random_erasing import RandomErasing
from timm.data.mixup import FastCollateMixup
from timm.data.transforms_factory import RandomResizedCropAndInterpolation
from timm.data.auto_augment import rand_augment_transform
from data.vtab import VTAB
# from torch.cuda.amp import autocast as amp_autocast

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x

class PrefetchLoader:

    def __init__(
            self,
            loader,
            mean=IMAGENET_INCEPTION_MEAN,
            std=IMAGENET_INCEPTION_STD,
            channels=3,
            fp16=False,
            re_prob=0.,
            re_mode='const',
            re_count=1,
            re_num_splits=0):

        mean = expand_to_chs(mean, channels)
        std = expand_to_chs(std, channels)
        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.:
            self.random_erasing = RandomErasing(probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x

def _worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def write(print_obj, log_file=None, end='\n'):
    print(print_obj, end=end)
    if log_file is not None:
        with open(log_file, 'a') as f:
            print(print_obj, end=end, file=f)

class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img

def create_transform(use_prefetcher=False, aug_type=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):

    if aug_type == 'VTAB':

        tfl = [
            transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]

    elif aug_type == 'FGVC_train':

        tfl = [
            RandomResizedCropAndInterpolation(224, interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
        ]


    elif aug_type =='FGVC_test':

        tfl = [
            transforms.Resize(248, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]

    elif aug_type == 'FGFS_train':

        tfl = [
            RandomResizedCropAndInterpolation(224, interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
        ]

        img_size_min = 224
        aa_params = dict(translate_const=int(img_size_min * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in mean]), interpolation=3)
        tfl += [rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params)]

    elif aug_type == 'FGFS_test':

        tfl = [
            transforms.Resize((256,256), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    elif aug_type == 'efficientnet_test':
        tfl = [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    else:
        raise NotImplementedError

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    transform = transforms.Compose(tfl)

    return transform

def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False

def create_loader(
        dataset,
        batch_size,
        is_training,
        re_prob,
        use_prefetcher,
        num_workers,
        collate_fn=None,
        fp16=False,
        persistent_workers=True,
        log_file=None
):


    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_training,
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if (is_training and not isinstance(dataset, VTAB)) else 0.
        write('prefetch_re_prob : {:.2f}'.format(prefetch_re_prob), log_file=log_file)
        loader = PrefetchLoader(
            loader,
            channels=3,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode='pixel',
            re_count=1,
            re_num_splits=0
        )

    return loader


@torch.no_grad()
def throughput(model,img_size=224,bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size=x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model.eval()
        for i in range(500):
            # with amp_autocast():
            model(x)
        torch.cuda.synchronize()

        count = 3000
        print("throughput averaged with {} times".format(count))
        tic1 = time.time()
        for i in range(count):
            # with amp_autocast():
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {count * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)