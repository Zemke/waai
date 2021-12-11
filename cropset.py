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
        img = Image.open("./labelling/source/" + doc["image"]).convert('RGB')
      x.append(T.ToTensor()(img))

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    with open(self.files[idx], 'r') as f:
      j = json.loads(f.read())
    for doc in j:
      img = Image.open("./labelling/source/" + doc["image"]).convert('RGB')
      boxes, labels = [], []
      for annot in doc["annotations"]:
        coord = annot["coordinates"]
        x, y, width, height = coord["x"], coord["y"], coord["width"], coord["height"]
        x1, y1, x2, y2 = x-width/2, y-height/2, x+width/2, y+height/2 
        boxes.append([x1, y1, x2, y2])
        labels.append(dataset.CLASSES.index(annot["label"]))
    return self.transform(img), {
      "boxes": torch.as_tensor(boxes, dtype=torch.float),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


class LolSet(Dataset):
  def __init__(self):
    # path,x1,y1,x2,y2,class
    self.df = pd.read_csv("./captureset/annot.csv")
    self.paths = [path for path in self.df["path"].unique()]

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    path = self.paths[idx]
    p_df = self.df[self.df["path"] == path]
    img = Image.open(path).convert('RGB')
    boxes, labels = [], []
    for e in p_df.iloc[:,1:].values:
      boxes.append(e[:-1])
      labels.append(e[-1])
    return self.transform(img), {
      "boxes": torch.as_tensor(boxes, dtype=torch.float),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


def collate_fn(batch):
  return tuple(zip(*batch))

def load(dataset, batch_size=4, shuffle=True):
  return DataLoader(
      dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

