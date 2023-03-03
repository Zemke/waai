#!/usr/bin/env python

import os
import sys
from collections import Counter
from contextlib import contextmanager

import torch
from torch.utils.data import \
    Dataset, DataLoader, random_split, Subset, WeightedRandomSampler, default_collate
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
WEAPONS = {"dynamite", "sheep"}
# TODO atm there's no diff btw girder part of map and user-deployed girder
ALWAYS = {'bg', 'blood', 'barrel', 'text', 'water', 'cloud', 'girder', 'worm', 'mine', 'puffs'}
MAPS = {'-beach', '-desert', '-farm', '-forest', '-hell', 'art', 'cheese', 'construction', 'desert', 'dungeon', 'easter', 'forest', 'fruit', 'gulf', 'hell', 'hospital', 'jungle', 'manhattan', 'medieval', 'music', 'pirate', 'snow', 'space', 'sports', 'tentacle', 'time', 'tools', 'tribal', 'urban'}
CLASSES = WEAPONS | ALWAYS | MAPS

if len(CLASSES) != sum(len(x) for x in [WEAPONS, ALWAYS, MAPS]):
  raise Exception("There are duplicate classes.")

AUG = {
  "water": [
    T.RandomResizedCrop((H, W), ratio=(2.,2.)),
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.1,.1)),
  ],
  "text": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
    T.RandomAffine(degrees=0, translate=(.1,.1)),
  ],
  "cloud": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
    T.RandomAffine(degrees=0, translate=(.1,.1)),
  ],
  "girder": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
    T.RandomAffine(degrees=0, translate=(.1,.1)),
  ],
  "barrel": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.1,.1)),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
  ],
  "blood": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
    T.RandomAffine(degrees=180, translate=(.1,.1)),
  ],
  "bg": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
    T.RandomAffine(degrees=0, translate=(.1,.1)),
  ],
  "mine": [
    T.RandomAffine(degrees=180, translate=(.1,.1)),
    T.RandomHorizontalFlip(p=.5),
  ],
  "worm": [
    T.RandomAffine(degrees=15, translate=(.2,.2)),
    T.RandomHorizontalFlip(p=.5),
  ],
  "dynamite": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "puffs": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
    T.RandomRotation(180),
  ],
  "sheep": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomResizedCrop((H, W), ratio=(1.,1.)),
  ],
}

for k in AUG.keys():
  if k not in CLASSES: print("augmentation for non-existing class", k)

AUG_MAP = [T.RandomHorizontalFlip(p=.4),]


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
    self._ctx_skip_normalize = False

    assert isinstance(classes, list)
    self.classes = classes

    df = pd.read_csv(self.annot_file)
    if len(unkn := df[~df["class"].isin(CLASSES)]["class"].unique()):
      raise Exception(f"unknown classes in annotations file: {unkn}")
    #df = pd.concat([df[df["class"] == c][:10] for c in self.classes])  # limit for debugging
    self.df = df[df["class"].isin(self.classes)]

  def compose_transform(self, clazz):
    transforms = [*TRANSFORMS]
    if not self._ctx_skip_augment:
      transforms.extend(augment(clazz))
    if self._ctx_skip_normalize:
      while 1:
        for i in range(len(transforms)):
          if isinstance(transforms[i], T.Normalize):
            del transforms[i]
            break
        else:
          break
    return T.Compose(transforms)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    file, clazz = self.df.iloc[idx]
    label = torch.tensor(self.classes.index(clazz))
    if self._ctx_skip_imread:
      return None, label
    img = read_image(os.path.join(self.img_dir, file), ImageReadMode.RGB)
    return self.compose_transform(clazz)(img), label, file

  @contextmanager
  def skip_imread(self, do=True):
    return self._skip_anything('imread', do)

  @contextmanager
  def skip_augment(self, do=True):
    return self._skip_anything('augment', do)

  @contextmanager
  def skip_normalize(self, do=True):
    return self._skip_anything('normalize', do)

  def _skip_anything(self, anything, do):
    attr = "_ctx_skip_" + anything
    try:
      setattr(self, attr, do)
      yield self
    finally:
      setattr(self, attr, not do)

  @staticmethod
  def collate_fn(batch):
    return default_collate(tuple(batch[:2]))


def splitset(ds):
  return random_split(ds, (.8, .2), torch.Generator().manual_seed(42))


def load(dataset, classes=None, weighted=False, **opts):
  if "batch_size" not in opts:
    opts["batch_size"] = int(os.getenv('BATCH', 8))
  if "collate_fn" not in opts:
    opts["collate_fn"] = MultiSet.collate_fn
  if "num_workers" not in opts and len(dataset) != bs:
    opts.update({"num_workers": 4, "persistent_workers": True})
  if weighted:
    if classes is None:
      raise Exception("classes must not be None for weighted sampling")
    cnt = 1 / torch.tensor(counts(dataset))
    with ds.dataset.skip_imread():
      weights = [cnt[v] for _,v in dataset]
    sampler = WeightedRandomSampler(weights, len(dataset))
    opts["sampler"] = sampler
  return DataLoader(dataset, **opts)


def classes(weapon=None):
  if weapon is None:
    return sorted(CLASSES)
  return sorted((CLASSES - WEAPONS) | {weapon})


def counts(ds, transl=False):
  with ds.dataset.skip_imread():
    c = Counter(l.item() for _,l in ds)
    c.update(i for i in range(len(ds.dataset.classes)))  # there could be missing classes
    if transl:
      return {ds.dataset.classes[v]: n for v,n in c.most_common()}
    return [c[i] for i in range(len(c))]


def transform(x):
  return T.Compose(TRANSFORMS)(x)


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import numpy as np
  from torchvision.utils import make_grid
  import torchvision.transforms.functional as F

  norm = '--normalize' in sys.argv
  for v in sys.argv:
    if v in CLASSES:
      clazzes = [v]
      break
  else:
    clazzes = sorted(set(classes()) - MAPS  - {'bg'})
  ds = MultiSet(clazzes)
  print(counts(ds, transl=True))
  BS = 256
  dl = load(ds, clazzes, weighted=True,
            batch_size=BS, num_workers=0,
            collate_fn=default_collate)
  with ds.skip_normalize(not norm):
    x, l, f = next(iter(dl))
    [print(f[i], l[i]) for i in range(BS)]
    with ds.skip_augment():
      o = torch.stack([ds.compose_transform(clazzes[l[i]])(read_image(os.path.join(ds.img_dir, f[i]), ImageReadMode.RGB)) for i in range(BS)])
  s = torch.cat((x,o),-1)
  plt.figure(figsize=(14,8))
  plt.xticks([])
  plt.yticks([])
  plt.tight_layout()
  plt.imshow(F.to_pil_image(make_grid(s, padding=8, nrow=BS//16, pad_value=1.)))
  plt.show()

