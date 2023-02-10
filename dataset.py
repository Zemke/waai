#!/usr/bin/env python

import os
from math import ceil, floor

import torch
from torch.utils.data import \
    Dataset, DataLoader, ConcatDataset, random_split, Subset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

from PIL import Image
import pandas as pd

import visual


CLASSES = ['bg', 'worm', 'mine', 'barrel', 'dynamite', 'sheep']
MEAN, STD = (.4134, .3193, .2627), (.3083, .2615, .2476)
MAPS = ['-beach', '-desert', '-farm', '-forest', '-hell', 'art', 'cheese', 'construction', 'desert', 'dungeon', 'easter', 'forest', 'fruit', 'gulf', 'hell', 'hospital', 'jungle', 'manhattan', 'medieval', 'music', 'pirate', 'snow', 'space', 'sports', 'tentacle', 'time', 'tools', 'tribal', 'urban']
C, W, H = 3, 30, 30
TRANSFORMS = [T.ToTensor(), T.Resize((H, W))]


class SingleSet(Dataset):
  @torch.no_grad()
  def __init__(self,
               single,
               df=None,
               annotations_file='./dataset/annot.csv',
               transform=None,
               img_dir='./dataset'):
    self.single = single
    self.img_dir = img_dir

    if df is None:
      clazz = CLASSES.index(single)
      df = pd.read_csv(annotations_file)
      df = df[df["label"] == clazz]
      df.loc[df["label"] == clazz, "label"] = int(bool(clazz))
      df.reset_index(inplace=True, drop=True)
    self.df = df

    self.transform = T.Compose([
      T.ToPILImage("RGB"),
      *TRANSFORMS,
      *([] if transform is None else transform)
      # TODO Normalize
    ])

  def augment(self, bg=False) -> ConcatDataset:
    # shared dataframe (df)
    augs = [
      self,
      SingleSet(self.single, df=self.df, transform=[T.RandomHorizontalFlip(p=1)]),
      SingleSet(self.single, df=self.df, transform=[
          T.RandomAffine(0, translate=(.2,.2)),
          T.RandomHorizontalFlip(p=.5),
        ]),
    ]
    if bg:
      augs.append(SingleSet('bg'))
    return ConcatDataset(augs)

  def count(self):
    return {self.single: cnt.item() for clazz,cnt in dict(self.df["label"].value_counts()).items()}

  @staticmethod
  def count_cum(dss: ConcatDataset):
    ab = [ds.count() for ds in dss.datasets]
    return {k: sum(x[k] if k in x else 0 for x in ab) for k in {x for y in ab for x in y}}

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    file, label = self.df.iloc[idx]
    return \
      self.transform(read_image(os.path.join(self.img_dir, file), ImageReadMode.RGB)), \
      torch.tensor(label, dtype=torch.float32)


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
               augment=False,
               annotations_file='./dataset/annot.csv',
               img_dir='./dataset'):
    self.img_dir = img_dir
    self.df = pd.read_csv(annotations_file)

    tt = T.Compose([
      T.ToPILImage("RGB"),
      T.Resize((30,30)),
      T.ToTensor(),
    ])

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
    file, label = self.df.iloc[idx]

    clazz = CLASSES[label]
    if clazz == 'worm':
      tt = [
        T.RandomAffine(0, translate=(.3,.3)),
        T.RandomHorizontalFlip(p=.5),
      ]
    elif clazz == 'mine':
      tt = [
        T.RandomRotation(40),
        T.RandomAffine(0, translate=(.2,.2)),
        T.RandomHorizontalFlip(p=.5),
      ]
    elif clazz == 'barrel':
      tt = [T.RandomAffine(0, translate=(.4,.4))]
    elif clazz == 'dynamite':
      tt = [T.RandomAffine(0, translate=(.2,.2))]
    elif clazz == 'sheep':
      tt = [
        T.RandomHorizontalFlip(p=.5),
        T.RandomAffine(0, translate=(.2,.2)),
      ]

    return \
      T.Compose([*self.transform, *tt])(self.imread(idx)), \
      torch.tensor(label, dtype=torch.float32)


def splitset(ds):
  splits = [ceil(len(ds)*.8), floor(len(ds)*.2)]
  train, test = random_split(ds, splits, torch.Generator().manual_seed(42))
  return train, test


def load(dataset, batch_size=8, shuffle=True):
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def transform(x):
  return T.Compose(TRANSFORMS)(x)

