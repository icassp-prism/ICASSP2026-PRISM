"""DataLoader builders for training and evaluation splits."""

import os

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from data.dataset import LLCMDataset, SYSUDataset
from data.dataset import RegDBDataset
from data.dataset import CMGroupDataset

from data.sampler import CrossModalityIdentitySampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import RandomIdentitySampler
from data.sampler import NormTripletSampler
from data.data_augmentation import RandomGrayscale

def collate_fn(batch):
    """Keep image paths as Python strings while stacking tensor fields."""

    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 4]
    data.insert(4, samples[4])
    return data

def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=4, split_num='4'):
    """Create the training loader with dataset-specific samples and sampler policy."""

    t = [T.Resize(image_size)]

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop:
        t.extend([T.Pad(padding, padding_mode='symmetric'), T.RandomCrop(image_size)])

    t.extend([T.ToTensor(), RandomGrayscale(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing(value='random'))

    transform = T.Compose(t)

    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform, split_num=split_num)
    elif dataset == 'llcm':
        train_dataset = LLCMDataset(root, mode='train', transform=transform)
    elif dataset == 'cmgroup_crop':
        train_dataset = CMGroupDataset(root, mode='train', transform=transform)

    # Sampler choice controls whether each mini-batch is balanced by modality or identity.
    assert sample_method in ['random', 'identity_uniform', 'identity_random', 'norm_triplet']
    if sample_method == 'identity_uniform':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_random':
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    else:
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)

    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader

def get_test_loader(dataset, root, batch_size, image_size, num_workers=4, split_num='4'):
    """Create gallery/query loaders without training-time augmentations."""

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform, split_num=split_num)
        query_dataset = RegDBDataset(root, mode='query', transform=transform, split_num=split_num)
    elif dataset == 'llcm':
        gallery_dataset = LLCMDataset(root, mode='gallery', transform=transform)
        query_dataset = LLCMDataset(root, mode='query', transform=transform)
    elif dataset == 'cmgroup_crop':
        gallery_dataset = CMGroupDataset(root, mode='gallery', transform=transform)
        query_dataset = CMGroupDataset(root, mode='query', transform=transform)

    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader
