#!/usr/bin/env python3

import os
import sys
from os import listdir
from os.path import isfile, join
from random import randrange

from PIL import Image, ImageFilter, ImageFile
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.transforms import v2

import norm
from rotate_fit import RandomRotationFit


ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES_WEAPONS = [
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
  "oldwoman",
  "chute",
  "petrol",
  #"drill",
  "select",
  "sheep",
  "skunk",
  #"surrender",
  #"teleport",
  "airstrike",
]
CLASSES_OTHER = ['bg', 'worm']
CLASSES = [*CLASSES_OTHER, *CLASSES_WEAPONS]

STD, MEAN = (0.339, 0.298, 0.32), (0.288, 0.238, 0.228)  # TODO
resize_table = {
  'mine': 8,
  'dynamite': 8,
  'barrel': 24,
  'worm': 22,  # TODO  worms have different sizes
  'jetpack': 13,
  'barrel': 30,
  'grenade': 14,
  'cluster': 14,
  "hhg": 18,
  "missile": 28,
  "pigeon": 18,
  "cow": 25,
  "mole": 20,
  "oldwoman": 25,
  "chute": 25,
  "petrol": 8,
  "select": 20,
  "sheep": 25,
  "skunk": 25,
  "airstrike": 20,
}

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
      v2.RandomHorizontalFlip(),
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
        custom_transforms = []
        if CLASSES[c] in [
          'mine',
          'grenade',
          'cluster',
          'hhg',
          'missile',
          'petrol',
        ]:
          custom_transforms.append(RandomRotationFit(180))
        elif CLASSES[c] == 'chute':
          custom_transforms.append(RandomRotationFit(30))
        elif CLASSES[c] == 'mole':
          custom_transforms.append(RandomRotationFit((-80,35)))
        elif c == CLASSES.index('barrel'):
          custom_transforms.append(
            v2.RandomChoice([
              v2.Resize((36, 24)),
              v2.Resize((33, 27)),
              v2.Resize(25),
              v2.Resize((30, 30)),
            ])
          )
        size = (resize_table)[CLASSES[c]]
        im2 = T.Compose([
          *self.transform_paste.transforms,
          v2.RandomResize(size-2,size+2),
          *custom_transforms,
          v2.ToPILImage(),
        ])(im2)
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
    dl = DataLoader(ds := DynamicSetPaste(batch_size), batch_size=batch_size)
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
    F.to_pil_image(denorm).show()
  elif sys.argv[1] == "smol":
    from torchvision.utils import make_grid
    from math import sqrt
    batch_size = 256
    dl = DataLoader(ds := DynamicSetPaste(batch_size), batch_size=batch_size)
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

