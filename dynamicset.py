#!/usr/bin/env python3

import os
import sys
from os import listdir
from os.path import isfile, join
from random import randrange

from PIL import Image, ImageFilter
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import v2

CLASSES = ['worm', 'mine', 'barrel', 'dynamite']
STD, MEAN = (.5, .5, .5), (.5, .5, .5)  # TODO

class DynamicSet(Dataset):
  def __init__(self, length):
    self.length = length;
    self.transform = T.Compose([
      T.PILToTensor(),
      v2.ToDtype(torch.float32, scale=True),
      # no norm in rcnn
      #T.Normalize(torch.tensor([0.8580, 0.4778, 0.2055]), torch.tensor([0.8040, 0.4523, 0.2539]))
    ])
    self.S = [30, 25, 40, 25]
    self.M = [Image.open('od/target/' + f).convert("RGBA") for f in listdir("od/target") if isfile(join("od/target", f)) and f.split(".")[-1] == "png"]
    self.W = [Image.open(c).convert("RGBA").resize((self.S[0], self.S[0])) for c in ['od/worms/' + f for f in listdir("od/worms") if isfile(join("od/worms", f)) and f.split(".")[-1] == "png"]]
    self.C = [Image.open(c).convert("RGBA").resize((self.S[t+1], self.S[t+1])) for t,c in enumerate([
      "od/weapons_alpha/mine.png",
      "od/weapons_alpha/barrel.png",
      "od/weapons_alpha/dynamite.png",
    ])]

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    back_im = self.M[randrange(len(self.M))].copy()
    wi = 0
    SP = 45
    height, width, _ = np.array(back_im).shape
    a = Image.new("RGBA", back_im.size, (0,0,0,0))
    boxes, labels = [], []
    for y in range(SP-30, height-SP, SP):
      for x in range(SP-30, width-SP, SP):
        im2 = self.W[randrange(len(self.W))] if (t := randrange(4)) == 0 else self.C[t-1]
        wi += 1
        a.paste(im2, (x1 := x+randrange(10), y1 := y+randrange(10)))
        labels.append(t)
        boxes.append([x1, y1, x1+self.S[t], y1+self.S[t]])
    back_im = Image.alpha_composite(back_im, a).convert("RGB")
    return self.transform(back_im), {
      "boxes": torch.as_tensor(boxes, dtype=torch.float16),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


def collate_fn(batch):
  return tuple(zip(*batch))

def load(dataset, batch_size):
  return DataLoader(
    dataset,
    num_workers=2, persistent_workers=True,
    batch_size=batch_size, collate_fn=collate_fn)

