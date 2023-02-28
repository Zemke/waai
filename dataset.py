#!/usr/bin/env python

import os
import sys
from collections import Counter
from contextlib import contextmanager

import torch
from torch.utils.data import \
    Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

from PIL import Image
import pandas as pd

import visual


STD, MEAN = (0.301, 0.262, 0.283), (0.401, 0.311, 0.256)
C, W, H = 3, 30, 30

TRANSFORMS = [
  T.ToPILImage("RGB"),
  T.Resize((H,W)),
  T.ToTensor(),
  T.Normalize(std=STD, mean=MEAN),
]

# all future weapons: "dynamite", "sheep", "bat", "torch", "bungee", "cluster", "drill", "hhg", "homing", "kamikaze", "bow", "cow", "napalm", "chute", "pigeon", "rope", "tele", "strike", "skunk", "axe", "jetpack", "gravity", "select"
# TODO atm there's no diff btw girder part of map and user-deployed girder
WEAPONS = {"dynamite", "sheep"}
ALWAYS = {"mine", 'phone', 'cloud', 'puffs', 'water', 'barrel', 'flag', 'worm', 'text', 'girder', 'healthbar', 'wind', 'blood'}
MAPS = {'-beach', '-desert', '-farm', '-forest', '-hell', 'art', 'cheese', 'construction', 'desert', 'dungeon', 'easter', 'forest', 'fruit', 'gulf', 'hell', 'hospital', 'jungle', 'manhattan', 'medieval', 'music', 'pirate', 'snow', 'space', 'sports', 'tentacle', 'time', 'tools', 'tribal', 'urban'}
CLASSES = WEAPONS | ALWAYS | MAPS

if len(CLASSES) != sum(len(x) for x in [WEAPONS, ALWAYS, MAPS]):
  raise Exception("There are duplicate classes.")

AUG = {
  "water": [
    T.RandomResizedCrop((H, W)),
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "text": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "cloud": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "girder": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "barrel": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "blood": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=180, translate=(.2,.2)),
  ],
  "bg": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "mine": [
    T.RandomAffine(degrees=180, translate=(.2,.2)),
    T.RandomHorizontalFlip(p=.5),
  ],
  "worm": [
    T.RandomAffine(degrees=20, translate=(.2,.2)),
    T.RandomHorizontalFlip(p=.5),
  ],
  "dynamite": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "puffs": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=180, translate=(.2,.2)),
  ],
  "sheep": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
    T.RandomResizedCrop((H, W)),
  ],
}

for k in AUG.keys():
  if k not in CLASSES: print("augmentation for non-existing class", k)

AUG_MAP = [T.RandomHorizontalFlip(p=.5),]


def augment(clazz):
  if clazz in MAPS:
    return AUG_MAP
  if clazz in AUG:
    return AUG[clazz]
  print("no augmentation for", clazz, file=sys.stderr)
  return []


class CaptureMultiSet(Dataset):
  def __init__(self, path):
    img = visual.load(path)
    div,mod = divmod(W, 3)
    if mod > 0:
      print(f"tiling missing last {mod} pixels on the right of each tile")
    self.tiles = visual.tile(img, kernel=H, stride=div)
    self.transform = T.Compose(TRANSFORMS)

  def __len__(self):
    return len(self.tiles)

  def __getitem__(self, idx):
    return self.transform(self.tiles[idx])


class MultiSet(Dataset):
  annot_file = './samples/annot.csv'
  img_dir = './samples'

  def __init__(self, classes):
    self.dataset = self
    self._ctx_skip_imread = False
    self._ctx_skip_augment = False

    assert isinstance(classes, list)
    self.classes = classes

    df = pd.read_csv(self.annot_file)
    if len(unkn := df[~df["class"].isin(CLASSES)]["class"].unique()):
      raise Exception(f"unknown classes in annotations file: {unkn}")
    #df = pd.concat([df[df["class"] == c][:10] for c in self.classes])  # limit for debugging
    self.df = df[df["class"].isin(self.classes)]

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    file, clazz = self.df.iloc[idx]
    label = torch.tensor(self.classes.index(clazz))
    if self._ctx_skip_imread:
      return None, label
    transforms = [*TRANSFORMS]
    if not self._ctx_skip_augment:
      transforms.extend(augment(clazz))
    # imshow augmentation
    #if clazz == 'mine':
    #  import matplotlib.pyplot as plt
    #  import numpy as np
    #  from time import sleep
    #  x = transforms(self._imread(idx)) * self.std[:, None, None] + self.mean[:, None, None]
    #  plt.imshow(x.permute((1,2,0)))
    #  plt.show()
    #  sleep(.5)
    img = read_image(os.path.join(self.img_dir, file), ImageReadMode.RGB)
    return T.Compose(transforms)(img), label

  @contextmanager
  def skip_imread(self, *args, **kwargs):
    return self._skip_anything('imread')

  @contextmanager
  def skip_augment(self, *args, **kwargs):
    return self._skip_anything('augment')

  def _skip_anything(self, anything):
    attr = "_ctx_skip_" + anything
    try:
      setattr(self, attr, True)
      yield self
    finally:
      setattr(self, attr, False)


def splitset(ds):
  return random_split(ds, (.8, .2), torch.Generator().manual_seed(42))


def load(dataset, classes=None, batch_size=None, weighted=False, shuffle=True):
  bs = int(os.getenv('BATCH', 8)) if batch_size is None else batch_size
  if bs != len(dataset):
    print('batch size is', bs)
  if weighted:
    if classes is None:
      raise Exception("classes must not be None for weighted sampling")
    cnt = 1 / torch.tensor(dataset.counts(dataset))
    weights = [cnt[v] for _,v in dataset]
    sampler = WeightedRandomSampler(weights, len(dataset))
    return DataLoader(dataset, batch_size=bs, sampler=sampler)
  else:
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle)


def classes(weapon=None):
  if weapon is None:
    return sorted(CLASSES)
  return sorted((CLASSES - WEAPONS) | {weapon})


def counts(ds, transl=False):
  with ds.dataset.skip_imread():
    c = Counter(l.item() for _,l in ds)
    c.update(i for i in range(len(ds.classes)))  # there could be missing classes
    if transl:
      return {ds.classes[v]: n for v,n in c.most_common()}
    return [c[i] for i in range(len(c))]


def transform(x):
  return T.Compose(TRANSFORMS)(x)

