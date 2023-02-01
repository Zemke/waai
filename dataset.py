#!/usr/bin/env python

import os
from math import ceil, floor

import torch
from torch.utils.data import \
    Dataset, DataLoader, ConcatDataset, random_split, Subset
import torchvision.transforms as T

from PIL import Image
import pandas as pd

import visual


CLASSES = ['bg', 'worm', 'mine', 'barrel', 'dynamite', 'sheep']
MEAN, STD = (.4134, .3193, .2627), (.3083, .2615, .2476)


class SingleSet(Dataset):
  @torch.no_grad()
  def __init__(self,
               single,
               label=1.,
               annotations_file='./dataset/annot.csv',
               transform=None,
               img_dir='./dataset'):
    df = pd.read_csv(annotations_file)
    self.label = label
    self.img_labels = df[df["label"].isin([CLASSES.index(single), -1])]
    self.img_dir = img_dir
    self.transform = T.Compose([
      T.ToTensor(),
      T.Resize((30,30)),
      *([] if transform is None else transform)
      # TODO Normalize
    ])

  def augment(self):
    return ConcatDataset([
      self,
      SingleSet('sheep', transform=[T.RandomHorizontalFlip(p=1)]),
      Subset(SingleSet('bg', label=0.), list(range(int(len(self) * 2 * 1.1))))
    ])

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = Image.open(img_path)
    # image = read_image(img_path) pytorch/vision#4181
    return self.transform(image), self.label


class CaptureSet(Dataset):
  @torch.no_grad()
  def __init__(self, path):
    img = visual.load(path)[150:-375, 150:-150,:]
    self.tiles = visual.tile(img, kernel=33)
    self.transform = T.Compose([
      T.ToTensor(),
      T.Resize((30,30)),
    ])

  @torch.no_grad()
  def __len__(self):
    return len(self.tiles)

  @torch.no_grad()
  def __getitem__(self, idx):
    return self.transform(self.tiles[idx])


class CaptureMultiSet(Dataset):
  @torch.no_grad()
  def __init__(self, path):
    img = visual.load(path)
    # TODO kernel is incompatible with singlenet
    self.tiles = visual.tile(img, kernel=30, stride=10)
    self.transform = T.Compose([
      T.ToTensor(),
      T.Resize((30,30)),
      T.Normalize(mean=MEAN, std=STD)
    ])


  @torch.no_grad()
  def __len__(self):
    return len(self.tiles)

  @torch.no_grad()
  def __getitem__(self, idx):
    return self.transform(self.tiles[idx])

class MultiSet(Dataset):

  def __init__(self,
               annotations_file='./dataset/annot.csv',
               img_dir='./dataset'):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir

    transform = T.Compose([T.Resize((30,30)), T.ToTensor(),])
    stck = []
    for idx in range(len(self.img_labels)):
      img_path = self.im_path(idx)
      image = Image.open(img_path).convert('RGB')
      stck.append(transform(image))
    self.std, self.mean = torch.std_mean(torch.stack(stck), (0,3,2))
    transform.transforms.append(T.Normalize(mean=self.mean, std=self.std))
    self.transform = transform

    self.augment()

  def augment(self):
    class AugmentDataset(Dataset):
      def __init__(self, transforms):
        self.imgs = []
        self.transforms = transforms
      def add(self, path, label, aug):
        self.imgs.append((path, label, aug))
      def __len__(self):
        return len(self.imgs)
      def __getitem__(self, idx):
        path, label, aug = self.imgs[idx]
        image = Image.open(path).convert('RGB')
        transform = T.Compose([aug, *self.transforms])
        #imshow augmentation
        #if label == CLASSES.index('mine'):
        #  import matplotlib.pyplot as plt
        #  plt.imshow(transform(image).permute((1,2,0)))
        #  plt.show()
        return transform(image), label

    aug_ds = AugmentDataset(self.transform.transforms)

    for i in range(len(self.img_labels)):
      label, path = self.img_labels.iloc[i, 1], self.im_path(i)
      clazz = CLASSES[label]
      if clazz == 'worm':
        aug_ds.add(path, label, T.RandomRotation(10))
        aug_ds.add(path, label, T.RandomAffine(0, translate=(.2,.2)))
        aug_ds.add(path, label, T.RandomHorizontalFlip(p=.5))
      elif clazz == 'mine':
        for _ in range(2):
          aug_ds.add(path, label, T.RandomRotation(40))
          aug_ds.add(path, label, T.RandomRotation(20))
        aug_ds.add(path, label, T.RandomAffine(0, translate=(.1,.1)))
        for i in range(4, 9, 2):
          aug_ds.add(path, label, T.Pad(i))
      elif clazz == 'barrel':
        for _ in range(4):
          aug_ds.add(path, label, T.RandomAffine(0, translate=(.1,.1)))
        for i in range(4, 11):
          aug_ds.add(path, label, T.Pad(i))
      elif clazz == 'dynamite':
        aug_ds.add(path, label, T.RandomAffine(0, translate=(.1,.1)))
      elif clazz == 'sheep':
        for _ in range(10):
          aug_ds.add(path, label, T.RandomHorizontalFlip(p=.5))
          aug_ds.add(path, label, T.RandomAffine(0, translate=(.2,.2)))

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

