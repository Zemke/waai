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

import norm

CLASSES_WEAPONS = [
  "mine",
  "barrel",
  "dynamite",
  "grenade",
  "hhg",
  "missile",
  "pigeon",
  "jetpack",
  "cow",
  "mole",
  "oldwoman",
  "chute",
  "petrol",
  "drill",
  "select",
  "sheep",
  "skunk",
  "surrender",
  "teleport",
  "airstrike",
]
CLASSES_OTHER = ['bg', 'worm']
CLASSES = [*CLASSES_OTHER, *CLASSES_WEAPONS]

STD, MEAN = (0.339, 0.298, 0.32), (0.288, 0.238, 0.228)

class DynamicSet(Dataset):
  def __init__(self, length):
    self.length = length;
    self.transform = T.Compose([
      T.PILToTensor(),
      v2.ToDtype(torch.float32, scale=True),
    ])
    self.M = [f for f in listdir("od/target") if isfile(join("od/target", f)) and f.split(".")[-1] == "png"]
    self.W = [Image.open(c).convert("RGBA") for c in ['od/worms/' + f for f in listdir("od/worms") if isfile(join("od/worms", f)) and f.split(".")[-1] == "png"]]
    self.C = [Image.open(c).convert("RGBA") for t,c in enumerate([
      f"od/weapons_alpha/{w}.png" for w in CLASSES_WEAPONS
    ])]
    self.transform_paste = T.Compose([
      T.PILToTensor(),
      v2.ToDtype(torch.float32, scale=True),
      # box fitting
      v2.Resize(60),
      v2.CenterCrop(50),
      # zoom and crop size
      v2.RandomResizedCrop(50, scale=(.4, 1.)),
      v2.RandomResize(20, 50),
      # augmentation
      v2.RandomHorizontalFlip(),
      v2.RandomApply([v2.RandomPosterize(2)], p=.3),
      v2.RandomApply([v2.GaussianBlur(7)], p=.3),
      v2.RandomApply([v2.RandomRotation(180)], p=.3),
      v2.ToPILImage(),
    ])

  def _get_img(self):
    c = randrange(len(CLASSES)-1) + 1
    return self.W[randrange(len(self.W))] if c == 1 else self.C[c-len(CLASSES_OTHER)], c

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    back_im = Image.open("od/target/" + self.M[randrange(len(self.M))]).convert("RGBA")
    a = Image.new("RGBA", back_im.size, (0,0,0,0))
    boxes, labels = [], []
    height, width, _ = np.array(back_im).shape
    SP, RND = 100, 50
    for y in range(0, height, SP):
      for x in range(0, width, SP):
        im2, c = self._get_img()
        im2 = self.transform_paste(im2)
        x1 = x + randrange(RND)
        y1 = y + randrange(RND)
        x2 = x1 + im2.width
        y2 = y1 + im2.height
        if x2 > width or y2 > height:
          continue
        a.paste(im2, (x1, y1))
        labels.append(c)
        boxes.append(z := [x1, y1, x2, y2])
    back_im = Image.alpha_composite(back_im, a).convert("RGB")
    return self.transform(back_im), {
      "boxes": torch.as_tensor(boxes, dtype=torch.float),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }

class DynamicSetPaste(DynamicSet):
  def __init__(self, length):
    super().__init__(length)
    self.HW = 30, 30
    self.transform = T.Compose([
      *self.transform_paste.transforms,
      v2.PILToTensor(),
      v2.Resize((self.HW[0],self.HW[1])),
      v2.ToDtype(torch.float, scale=True),
    ])

  def __getitem__(self, idx):
    im2, c = self._get_img()
    return \
      self.transform(im2.convert("RGB")), \
      torch.as_tensor(c, dtype=torch.int64)


class DynamicOverfitSet(DynamicSet):

  def __init__(self, length):
    super().__init__(length)
    self.D = [None] * length

  def __getitem__(self, idx):
    if self.D[idx] is None:
      self.D[idx] = super().__getitem__(idx)
      v2.ToPILImage()(self.D[idx][0]).save('overfit/' + str(idx) + '.png')
    return self.D[idx][0], self.D[idx][1]


def collate_fn(batch):
  return tuple(zip(*batch))

def load(dataset, batch_size, pin_memory=False):
  workers = dict(
    num_workers=(wrks := int(os.getenv("WORKERS", 2))),
    persistent_workers=wrks > 0,
  )
  print('workers', workers, 'pin_memory:', pin_memory)
  return DataLoader(
    dataset,
    pin_memory=pin_memory,
    **workers,
    batch_size=batch_size,
    collate_fn=collate_fn)


if __name__ == "__main__":
  if sys.argv[1] == 'norm':
    batch_size = 10_000
    dl = DataLoader(ds := DynamicSetPaste(batch_size), batch_size=batch_size)
    norm.exec(ds, dl, ds.HW)
  elif sys.argv[1] == "big":
    from torchvision.utils import draw_bounding_boxes
    import torchvision.transforms.functional as F
    transed, bl = DynamicSet(1)[1]
    boxes = bl["boxes"]
    labels = bl["labels"]
    denorm = (transed*255).to(torch.uint8)
    bb = draw_bounding_boxes(denorm, boxes, [CLASSES[l] for l in labels])
    F.to_pil_image(bb).show()
  elif sys.argv[1] == "smol":
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as F
    from math import sqrt
    batch_size = 256
    dl = DataLoader(ds := DynamicSetPaste(batch_size), batch_size=batch_size)
    x = F.normalize(next(iter(dl))[0], MEAN, STD)
    F.to_pil_image(make_grid(x, nrow=int(sqrt(batch_size)))).show()

