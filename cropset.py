#!/usr/bin/env python

import os 
import json
from PIL import Image

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import dataset


class CropSet(Dataset):
  def __init__(self):
    self.files = []
    self.transform = T.Compose([T.ToTensor()])
    with os.scandir("./labelling/target") as it:
      for entry in it:
        if entry.is_file() and entry.name.lower().endswith('.json'):
          self.files.append(entry.path)

    x = []
    for idx in range(len(self.files)):
      with open(self.files[idx], 'r') as f:
        j = json.loads(f.read())
      for doc in j:
        img = Image.open(os.path.join("captures", doc["image"])).convert('RGB')
      x.append(T.ToTensor()(img))

  def augment(self):
    class AugmentSet(Dataset):
      def __init__(self, default_transforms):
        self.files = []
        self.def_transf = default_transforms
      def add(self, f, aug_transf):
        self.files.append((f, aug_transf))
      def __len__(self):
        return len(self.files)
      def __getitem__(self, idx):
        return CropSet._getitem(
          self.files[idx][0],
          T.Compose([*self.def_transf.transforms, self.files[idx][1]]))

    ds = AugmentSet(self.transform)
    for f in self.files:
      ds.add(f, T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
    return self + ds

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    return self._getitem(self.files[idx], self.transform)

  @staticmethod
  def _getitem(file, transform):
    with open(file, 'r') as f:
      j = json.loads(f.read())
    for doc in j:
      img = Image.open(os.path.join("captures", doc["image"])).convert('RGB')
      boxes, labels = [], []
      for annot in doc["annotations"]:
        coord = annot["coordinates"]
        x, y, width, height = coord["x"], coord["y"], coord["width"], coord["height"]
        x1, y1, x2, y2 = x-width/2, y-height/2, x+width/2, y+height/2 
        boxes.append([x1, y1, x2, y2])
        labels.append(dataset.CLASSES.index(annot["label"]))
    transed = transform(img)
    # show augmentation
    #if len(transform.transforms) > 1:
    #  import matplotlib.pyplot as plt
    #  plt.imshow(transed.permute((1,2,0)))
    #  plt.show()
    return transed, {
      "boxes": torch.as_tensor(boxes, dtype=torch.float),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


def collate_fn(batch):
  return tuple(zip(*batch))


def load(dataset, batch_size=4, shuffle=True):
  return DataLoader(
      dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

