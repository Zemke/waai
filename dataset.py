#!/usr/bin/env python

import os
from math import ceil, floor
from collections import Counter

import torch
from torch.utils.data import \
    Dataset, DataLoader, ConcatDataset, \
    random_split, Subset, WeightedRandomSampler
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

from PIL import Image
import pandas as pd

import visual


MEAN, STD = (.4134, .3193, .2627), (.3083, .2615, .2476)
C, W, H = 3, 30, 30

TRANSFORM = [
  T.ToPILImage("RGB"),
  T.Resize((H,W)),
  T.ToTensor(),
  T.Normalize(mean=MEAN, std=STD),
]

# TODO atm there's no diff btw girder part of map and user-deployed girder
WEAPONS = {"dynamite", "sheep", "bat", "torch", "bungee", "cluster", "drill", "hhg", "homing", "kamikaze", "bow", "cow", "napalm", "chute", "pigeon", "rope", "tele", "strike", "skunk", "axe", "jetpack", "gravity", "select"}
ALWAYS = {"mine", 'phone', 'cloud', 'puffs', 'water', 'barrel', 'flag', 'worm', 'text', 'girder', 'healthbar', 'wind', 'blood'}
MAPS = {'-beach', '-desert', '-farm', '-forest', '-hell', 'art', 'cheese', 'construction', 'desert', 'dungeon', 'easter', 'forest', 'fruit', 'gulf', 'hell', 'hospital', 'jungle', 'manhattan', 'medieval', 'music', 'pirate', 'snow', 'space', 'sports', 'tentacle', 'time', 'tools', 'tribal', 'urban'}
CLASSES = WEAPONS | ALWAYS | MAPS

if len(CLASSES) != sum(len(x) for x in [WEAPONS, ALWAYS, MAPS]):
  raise Exception("There are duplicate classes.")


class CaptureMultiSet(Dataset):
  def __init__(self, path):
    img = visual.load(path)
    self.tiles = visual.tile(img, kernel=H, stride=W//3)
    self.transform = T.Compose(TRANSFORM)

  def __len__(self):
    return len(self.tiles)

  def __getitem__(self, idx):
    return self.transform(self.tiles[idx])


class MultiSet(Dataset):

  def __init__(self,
               weapon=None,
               augment=False,
               annotations_file='./dataset/annot.csv',
               img_dir='./dataset'):
    self.img_dir = img_dir

    df = pd.read_csv(annotations_file)
    if len(unkn := df[~df["class"].isin(CLASSES)]["class"].unique()):
      raise Exception(f"unknown classes: {unkn}")
    if weapon is not None:
      assert weapon in WEAPONS or weapon == 'mine'
      # filter all other weapons
      df = df[~df["class"].isin([w for w in WEAPONS if w != weapon])]
    self.df = df
    self.classes = sorted(df["class"].unique())

    tt = T.Compose(TRANSFORM[:-1])
    stck = torch.stack([tt(self.imread(i)) for i in range(len(self.df))])
    self.std, self.mean = torch.std_mean(stck, (0,3,2))
    tt.transforms.append(T.Normalize(mean=self.mean, std=self.std))
    self.transform = tt

  def imread(self, idx):
    return read_image(
      os.path.join(self.img_dir, self.df.iloc[idx]["file"]),
      ImageReadMode.RGB)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    clazz = self.df.iloc[idx]["class"]
    transform = T.Compose([*self.transform.transforms, *self._augments(clazz)])
    # imshow augmentation
    #if clazz == 'mine':
    #  import matplotlib.pyplot as plt
    #  import numpy as np
    #  from time import sleep
    #  x = transform(self.imread(idx)) * self.std[:, None, None] + self.mean[:, None, None]
    #  plt.imshow(x.permute((1,2,0)))
    #  plt.show()
    #  sleep(1)
    return \
      transform(self.imread(idx)), \
      torch.tensor(self.classes.index(clazz))

  def _augments(self, clazz):
    if clazz == 'worm':
      tt = [
        T.RandomAffine(degrees=0, translate=(.3,.3)),
        T.RandomHorizontalFlip(p=.5),
      ]
    elif clazz == 'mine':
      tt = [
        T.RandomRotation(40),
        T.RandomAffine(degrees=0, translate=(.2,.2)),
        T.RandomHorizontalFlip(p=.5),
      ]
    elif clazz == 'barrel':
      tt = [T.RandomAffine(degrees=0, translate=(.4,.4))]
    elif clazz == 'dynamite':
      tt = [T.RandomAffine(degrees=0, translate=(.2,.2))]
    elif clazz == 'sheep':
      tt = [
        T.RandomHorizontalFlip(p=.5),
        T.RandomAffine(degrees=0, translate=(.2,.2)),
      ]
    elif clazz in MAPS:
      tt = [T.RandomHorizontalFlip(p=.5)]
    elif clazz == "debris":
      tt = [
        T.RandomPerspective(distortion_scale=.2, p=.5),
        T.RandomAffine(degrees=180, translate=(.2,.2)),
      ]
    elif clazz == "water" or clazz == "cloud":
      tt = [
        T.RandomHorizontalFlip(p=.5),
        T.RandomAffine(degrees=0, translate=(.2,.2)),
      ]
    elif clazz == "rope":
      tt = [
        T.RandomAffine(degrees=180, translate=(.3,.3)),
      ]
    elif clazz == "crate":
      tt = [
        T.RandomAffine(degrees=30, translate=(.5,.5)),
      ]
    elif clazz in ["text", "crate", "puffs", "phone", "healthbar", "girder", "flag"]:
      tt = [
        T.RandomAffine(degrees=0, translate=(.1,.1)),
      ]
    else:
      tt = []
    return tt


def splitset(ds):
  splits = [ceil(len(ds)*.8), floor(len(ds)*.2)]
  train, test = random_split(ds, splits, torch.Generator().manual_seed(42))
  return train, test


def load(dataset, classes, batch_size=None, weighted=False, shuffle=True):
  bs = int(os.getenv('BATCH', 8)) if batch_size is None else batch_size
  if bs != len(dataset):
    print('batch size is', bs)
  if weighted:
    cnt = 1 / torch.tensor(counts(classes, dataset))
    weights = [cnt[v] for _,v in dataset]
    sampler = WeightedRandomSampler(weights, len(dataset))
    return DataLoader(dataset, batch_size=bs, sampler=sampler)
  else:
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle)


def counts(classes, ds, transl=False):
  c = Counter([l.item() for _,l in ds])
  if transl:
    return {classes[v]: n for v,n in c.most_common()}
  return [c[i] for i in range(len(c))]


def transform(x):
  return T.Compose(TRANSFORM)(x)

