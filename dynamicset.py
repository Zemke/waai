#!/usr/bin/env python3

import os
import sys
from os import listdir
from os.path import isfile
from random import randrange

from PIL import Image, ImageFilter, ImageFile
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from torchvision.transforms import v2

import norm
from rotate_fit import RandomRotationFit


ImageFile.LOAD_TRUNCATED_IMAGES = True


FG_CLASSES = [c for c in [
  "worm",
  "mine",
  "barrel",
  "dynamite",
  "grenade",
  "cluster",
  "hhg",
  "missile",
  "pigeon",
  "jetpack",
  "cow",
  "mole",
  "granny",
  "chute",
  "petrol",
  "select",
  "sheep",
  "skunk",
  "airstrike",
] if (clz := os.getenv("CLS")) is None or c in clz.split(",")]
CLASSES = ['bg', *FG_CLASSES]
STD, MEAN = (0.339, 0.298, 0.32), (0.288, 0.238, 0.228)

class DynamicSet(Dataset):
  def __init__(self, length):
    self.length = length;
    self.transform = v2.Compose([
      v2.PILToTensor(),
      v2.ToDtype(torch.float32, scale=True),
    ])
    self.M = [Image.open("maps/" + f).convert("RGBA") for f in listdir("maps") if isfile("maps/" + f) and f.split(".")[-1] == "png"]
    self.T = {
      "worm_staring": [self._urr(28)],
      "worm_stepping": [self._urr(30)],
      "worm_sliding": [self._urr(28)],
      "worm_enter2": [self._urr(20)],
      "worm_sick": [self._urr(24)],
      "worm_hovering": [self._urr(18)],
      "worm_readying": [self._urr(24)],
      "worm_wincing": [self._urr(25)],
      "worm_scratching": [self._urr(24)],
      "worm_lookdown": [self._urr(24)],
      "worm_jetpacking": [self._urr(18)],
      "worm_falling": [
        self._urr(32),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR)
      ],
      "worm_doublebackspace": [
        self._urr(28),
        RandomRotationFit(22, interpolation=InterpolationMode.BILINEAR)
      ],
      "worm_slope": [self._urr(20)],
      "worm_roping": [
        self._urr(26),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR)
      ],
      "worm_default": [self._urr(28)],
      'mine': [
        self._urr(8),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR),
      ],
      'dynamite': [self._urr(8)],
      'jetpack': [self._urr(13)],
      'barrel': [
        self._urr(30),
        v2.RandomChoice([
          v2.Resize((36, 24)),
          v2.Resize((33, 27)),
          v2.Resize(25),
          v2.Resize((30, 30)),
        ])
      ],
      'grenade': [
        self._urr(14),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR)
      ],
      'cluster': [
        self._urr(14),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR)
      ],
      "hhg": [
        self._urr(18),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR),
      ],
      "missile": [
        self._urr(28),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR)
      ],
      "pigeon": [self._urr(18)],
      "cow": [self._urr(25)],
      "mole": [
        self._urr(20),
        RandomRotationFit((-80,35), interpolation=InterpolationMode.BILINEAR)
      ],
      "granny": [self._urr(25)],
      "chute": [
        self._urr(25),
        RandomRotationFit(30, interpolation=InterpolationMode.BILINEAR)
      ],
      "petrol": [
        self._urr(8),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR)
      ],
      "select": [
        self._urr(24),
        RandomRotationFit(180, interpolation=InterpolationMode.BILINEAR)
      ],
      "sheep": [self._urr(25)],
      "skunk": [self._urr(25)],
      "airstrike": [self._urr(20)],
    }
    self.F = [Image.open(f"objects/{f}.png").convert("RGBA") for f in self.T.keys()]

  def _urr(self, size, uncertainty=2):
    """uncertainty random resize"""
    return v2.RandomResize(min_size=size-uncertainty, max_size=size+uncertainty)

  def _get_img(self):
    while True:
      idx_T = randrange(len(self.T))
      key_T = [*self.T.keys()][idx_T]
      if key_T.split("_")[0] in CLASSES:
         break
    return \
      v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(),
        *self.T[key_T],
        v2.ToPILImage(),
      ])(self.F[idx_T]), \
      CLASSES.index(key_T.split("_")[0])

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    back_im = self.M[randrange(len(self.M))]
    a = Image.new("RGBA", back_im.size, (0,0,0,0))
    boxes, labels = [], []
    height, width, _ = np.array(back_im).shape
    SP, RND = 100, 50
    for y in range(0, height, SP):
      for x in range(0, width, SP):
        paste_img, c = self._get_img()
        x1, y1 = x + randrange(RND), y + randrange(RND)
        x2, y2 = x1 + paste_img.width, y1 + paste_img.height
        if x2 > width or y2 > height:
          continue
        a.paste(paste_img, (x1, y1))
        labels.append(c)
        boxes.append([x1, y1, x2, y2])
    back_im = Image.alpha_composite(back_im, a).convert("RGB")
    return self.transform(back_im), {
      "boxes": torch.as_tensor(boxes, dtype=torch.float),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }

class DynamicPasteSet(DynamicSet):
  def __init__(self, length):
    super().__init__(length)
    self.HW = 30, 30
    self.transform = v2.Compose([
      v2.PILToTensor(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Resize((self.HW[0],self.HW[1])),
      v2.ToDtype(torch.float, scale=True),
    ])

  def __getitem__(self, idx):
    paste_img, c = self._get_img()
    return \
      self.transform(paste_img.convert("RGB")), \
      torch.as_tensor(c, dtype=torch.int64)


class DynamicMemSet(DynamicSet):
  MAX_LENGTH = 10_000

  def __init__(self, length=None):
    length = self.MAX_LENGTH if length is None else length
    super().__init__(length)
    assert length <= self.MAX_LENGTH
    print(length)
    self.Y = torch.load('mem/mem.pt')[:length]

  def __getitem__(self, idx):
    return self.transform(Image.open(f"mem/{idx}.png")), self.Y[idx]


class DynamicGenSet(DynamicSet):

  def __init__(self, length):
    super().__init__(length)
    # TODO get std and mean from pastes?

  def __getitem__(self, idx):
    img, y = super().__getitem__(idx)
    return img, y, idx


class DynamicRandSet(DynamicSet):

  def __init__(self, length):
    super().__init__(length)

  def __getitem__(self, idx):
    return torch.randn((3, 696, 1920)), {
      'labels': torch.randint(0, len(CLASSES), (133,)).to(torch.int64),
      'boxes': torch.tensor([1,2,3,4]).repeat(133,1).to(torch.float)
    }


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
    dl = DataLoader(ds := DynamicPasteSet(batch_size), batch_size=batch_size)
    norm.exec(ds, dl, ds.HW)
  elif sys.argv[1] == "big":
    from torchvision.utils import draw_bounding_boxes
    transed, bl = DynamicSet(1)[1]
    boxes = bl["boxes"]
    labels = bl["labels"]
    denorm = (transed*255).to(torch.uint8)
    if os.getenv("BB") != "0":
      bb = draw_bounding_boxes(
        denorm, boxes, [CLASSES[l] for l in labels], colors='cyan')
      denorm = bb
    from pprint import pprint
    pprint({CLASSES[c]: len(labels[labels == c]) for c in range(len(CLASSES))})
    print(len(labels))
    F.to_pil_image(denorm).show()
  elif sys.argv[1] == "smol":
    from torchvision.utils import make_grid
    from math import sqrt
    batch_size = 256
    dl = DataLoader(ds := DynamicPasteSet(batch_size), batch_size=batch_size)
    x = next(iter(dl))[0]
    if os.getenv("NORM") != "0":
      x = F.normalize(x, MEAN, STD)
    F.to_pil_image(make_grid(x, nrow=int(sqrt(batch_size)))).show()
  elif sys.argv[1] == "memgen":
    dl = load(ds := DynamicGenSet(DynamicMemSet.MAX_LENGTH), batch_size=10)
    Y = [None] * len(ds)
    for imgs, y, idx in tqdm(dl):
      for i in range(len(imgs)):
        F.to_pil_image(imgs[i]).save('mem/' + str(idx[i]) + '.png')
        Y[idx[i]] = y[i]
    torch.save(Y, 'mem/mem.pt')
  elif sys.argv[1] == "memshow":
    from torchvision.utils import draw_bounding_boxes
    ds = DynamicMemSet()
    for i in range(0, len(ds), (len(ds)-2)//10):
      img, y = ds[i]
      boxes = y["boxes"]
      labels = y["labels"]
      denorm = (img*255).to(torch.uint8)
      bb = draw_bounding_boxes(
        denorm, boxes, [CLASSES[l] for l in labels], colors='cyan')
      denorm = bb
      F.to_pil_image(denorm).show()

