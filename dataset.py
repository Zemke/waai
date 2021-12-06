#!/usr/bin/env python

import os
from math import ceil, floor

import torch
from torch.utils.data import \
    Dataset, DataLoader, ConcatDataset, random_split
from torchvision.transforms import \
    Compose, ToTensor, Resize, \
    RandomRotation, RandomAffine, RandomHorizontalFlip, Pad, \
    functional as F

import numpy as np
from PIL import Image
import pandas as pd

import visual

CLASSES = ['worm', 'mine', 'barrel', 'dynamite', 'neg']


class SingleSet(Dataset):
  @torch.no_grad()
  def __init__(self,
               annotations_file='./dataset/annot.csv',
               img_dir='./dataset'):
    df = pd.read_csv(annotations_file)
    self.img_labels = df[df["label"].isin([CLASSES.index('dynamite'), -1])]
    self.img_dir = img_dir

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = Image.open(img_path)
    # image = read_image(img_path) pytorch/vision#4181
    return F.to_tensor(image), float(self.img_labels.iloc[idx, 1])


class CaptureSet(Dataset):
  @torch.no_grad()
  def __init__(self, path):
    img = visual.load(path)[50:-75, 50:-50,:]
    self.tiles = visual.tile(img)

  @torch.no_grad()
  def __len__(self):
    return len(self.tiles)

  @torch.no_grad()
  def __getitem__(self, idx):
    return F.to_tensor(self.tiles[idx])


class CaptureMultiSet(Dataset):
  @torch.no_grad()
  def __init__(self, path):
    img = visual.load(path)
    # TODO kernel=30 is incompatible with singlenet
    self.tiles = visual.tile(img, kernel=30)

  @torch.no_grad()
  def __len__(self):
    return len(self.tiles)

  @torch.no_grad()
  def __getitem__(self, idx):
    return F.to_tensor(self.tiles[idx])

class MultiSet(Dataset):

  def __init__(self,
               annotations_file='./dataset/annot.csv',
               img_dir='./dataset'):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = Compose([ToTensor(), Resize((30,30))])
    self.augment()

  def augment(self):
    class AugmentDataset(Dataset):
      def __init__(self):
        self.imgs = []
        self.to_tensor = ToTensor()
        self.resize = Resize((30,30))
      def add(self, path, label, transform):
        self.imgs.append([path, label, transform])
      def __len__(self):
        return len(self.imgs)
      def __getitem__(self, idx):
        path, label, transform = self.imgs[idx]
        image = Image.open(path).convert('RGB')
        compose = Compose([self.to_tensor, transform, self.resize])
        return compose(image), label

    aug_ds = AugmentDataset()

    for i in range(len(self.img_labels)):
      label, path = self.img_labels.iloc[i, 1], self.im_path(i)
      clazz = CLASSES[label]
      if clazz == 'worm':
        aug_ds.add(path, label, RandomRotation(10))
        aug_ds.add(path, label, RandomAffine(0, translate=(.2,.2)))
        aug_ds.add(path, label, RandomHorizontalFlip(p=.5))
      elif clazz == 'mine':
        for _ in range(3):
          aug_ds.add(path, label, RandomRotation(40))
        for _ in range(4):
          aug_ds.add(path, label, RandomRotation(20))
        for _ in range(3):
          aug_ds.add(path, label, RandomAffine(0, translate=(.1,.1)))
        for i in range(4, 9, 2):
          aug_ds.add(path, label, Pad(i, padding_mode='edge'))
      elif clazz == 'barrel':
        for _ in range(4):
          aug_ds.add(path, label, RandomAffine(0, translate=(.1,.1)))
        for i in range(4, 11):
          aug_ds.add(path, label, Pad(i))
      elif clazz == 'dynamite':
        aug_ds.add(path, label, RandomAffine(0, translate=(.1,.1)))

    return self + aug_ds

  def im_path(self, idx):
    return os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = self.im_path(idx)
    image = Image.open(img_path).convert('RGB')
    # image = read_image(img_path) pytorch/vision#4181
    return self.transform(image), self.img_labels.iloc[idx, 1]


def splitset(ds):
  splits = [ceil(len(ds)*.8), floor(len(ds)*.2)]
  train, test = random_split(ds, splits, torch.Generator().manual_seed(42))
  return train, test


def load(dataset, batch_size=8, shuffle=True):
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

