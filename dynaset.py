#!/usr/bin/env python

import os
from math import ceil, floor

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import \
  Compose, ToTensor, Normalize, Resize

from PIL import Image
import pandas as pd

import visual


class DynaSet(Dataset):
  @torch.no_grad()
  def __init__(self,
               annotations_file='./dataset/annot.csv',
               img_dir='./dataset',
               standardized=False):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = Compose([ToTensor(), Resize((25,25))])
    if not standardized:
      stdz = DynaSet(
          annotations_file=annotations_file,
          img_dir=img_dir,
          standardized=True)
      std, mean = torch.std_mean(
          torch.stack([i[0] for i in stdz]), (0,2,3))
      self.transform.transforms.append(Normalize(std=std, mean=mean))

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = Image.open(img_path)
    # image = read_image(img_path) pytorch/vision#4181
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
      image = self.transform(image)
    return image, float(label)


def splitset(ds):
  splits = [ceil(len(ds)*.9), floor(len(ds)*.1)]
  train, test = random_split(ds, splits, torch.Generator().manual_seed(42))
  return train, test


def load(dataset, batch_size=4, shuffle=True):
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CaptureSet(Dataset):
  @torch.no_grad()
  def __init__(self, path):
    img = visual.load(path)[50:-75, 50:-50,:]
    self.tiles = visual.tile(img)
    self.tensors = [ToTensor()(i) for i in self.tiles]
    std, mean = torch.std_mean(torch.stack(self.tensors), (0,2,3))
    self.norm = Normalize(std=std, mean=mean)

  @torch.no_grad()
  def __len__(self):
    return len(self.tiles)

  @torch.no_grad()
  def __getitem__(self, idx):
    return self.norm(self.tensors[idx])

