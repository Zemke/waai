#!/usr/bin/env python

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
import pandas as pd


class DynaSet(Dataset):
  def __init__(self,
               annotations_file='./dataset/annot.csv',
               img_dir='./dataset',
               transform=None,
               target_transform=float,
               standardized=False):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    if not standardized:
      stdz = DynaSet(
          annotations_file=annotations_file,
          img_dir=img_dir,
          transform=transform,
          target_transform=target_transform,
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
    if self.target_transform:
      label = self.target_transform(label)
    return image, label


def load():
  return DataLoader(
    DynaSet(transform=Compose([ToTensor(), Resize((25,25))])),
    batch_size=4, shuffle=True)

