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
from torchvision.transforms import InterpolationMode

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
CLASSES = ["bg", "dynamite", "mine", "barrel", "worm", "sheep", "girder"]


AUG = {
  "bg": [],  # TODO bg: horizontal flip? affine? resizedcrop?
  "girder": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W), scale=(.8, 1.), ratio=(.3, 1.5)),
    T.RandomAffine(degrees=0, translate=(.3,.3)),
    T.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
  ],
  "barrel": [
    T.ColorJitter(brightness=(1.,3.)),
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
    T.RandomResizedCrop((H, W), scale=(.8, 1.), ratio=(.3, 1.5)),
  ],
  "mine": [
    T.RandomAffine(degrees=180, translate=(.2,.2), interpolation=InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W), scale=(.5, 1.)),
  ],
  "worm": [
    T.RandomAffine(degrees=15, translate=(.2,.2), interpolation=InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(p=.5),
    T.RandomResizedCrop((H, W), scale=(.3, 1.)),
  ],
  "dynamite": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.1,.25), interpolation=InterpolationMode.BILINEAR),
    T.RandomResizedCrop((H, W), scale=(.5,1.), ratio=(.3,2.)),
  ],
  "sheep": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomResizedCrop((H, W), scale=(.3, 1.)),
    T.RandomAffine(degrees=0, translate=(.3,.3), interpolation=InterpolationMode.BILINEAR),
  ],
}

for k in AUG.keys():
  if k not in CLASSES: print("augmentation for non-existing class", k)


def augment(clazz):
  if clazz in AUG:
    return [T.RandomApply(AUG[clazz], p=.5)]
  print("no augmentation for", clazz, file=sys.stderr)
  return []


class CaptureSet(Dataset):
  transform = T.Compose(TRANSFORMS)

  def __init__(self, source_dir, target_dir):
    self.div, self.mod = divmod(W, 3)
    if self.mod > 0:
      print(f"tiling missing last {mod} pixels on the right of each tile")

    imgs = []
    with os.scandir(source_dir) as it:
      for entry in it:
        if entry.is_file() and entry.name.lower().endswith('.png'):
          imgs.append(entry.path)
    if not os.path.isdir(target_dir):
      print(f"{target_dir} not found", file=sys.stderr)
      exit(4)
    self.imgs = imgs

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    capture = visual.load(self.imgs[idx])
    tiles = visual.tile(capture, kernel=W, stride=self.div)
    orig, trans = zip(*[(t, self.transform(t)) for t in tiles])
    return torch.stack(trans), orig

  @staticmethod
  def load(dataset):
    assert isinstance(dataset, CaptureSet)
    return DataLoader(
      dataset,
      batch_size=None,
      batch_sampler=None,
      collate_fn=CaptureSet._collate_fn,
      num_workers=4,
      persistent_workers=True)

  @staticmethod
  def _collate_fn(b):
    return b


class MultiSet(Dataset):
  annot_file = './samples/annot.csv'
  img_dir = './samples'

  def __init__(self):
    self.dataset = self
    self._ctx_skip_imread = False
    self._ctx_skip_augment = False
    self._ctx_skip_normalize = False

    df = pd.read_csv(self.annot_file, header=None)
    df[1] = df.iloc[:,0].str.split('/', n=1, expand=True)[0]
    df.columns = ["file", "class"]
    if len(unkn := df[~df["class"].isin(CLASSES)]["class"].unique()):
      raise Exception(f"unknown classes in annotations file: {unkn}")
    #df = pd.concat([df[df["class"] == c][:10] for c in CLASSES])  # limit for debugging
    self.df = df

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
    label = torch.tensor(CLASSES.index(clazz))
    if self._ctx_skip_imread:
      return None, label, file
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
    return default_collate([(x,l) for x,l,_ in batch])

  @staticmethod
  def load(ds, weighted=False, **opts):
    if not isinstance(ds.dataset, MultiSet):
      raise Exception("load is only for MultiSet or Subset of it")
    if "batch_size" not in opts:
      opts["batch_size"] = int(os.getenv('BATCH', 8))
    if "collate_fn" not in opts and isinstance(ds.dataset, MultiSet):
      opts["collate_fn"] = MultiSet.collate_fn
    if "num_workers" not in opts and len(ds) != opts["batch_size"]:
      opts.update({"num_workers": 4, "persistent_workers": True})
    if "pin_memory" not in opts and os.getenv("GPU") == '1':
      opts["pin_memory"] = True
    if weighted:
      # TODO cache weights
      print('weighing classes')
      with ds.dataset.skip_imread():
        bg_c = Counter(f.split("/")[1] for _,l,f in ds if l == 0)
      cnt = 1 / torch.tensor(counts(ds))
      bg_mul = (len(bg_c) - (maps_mul := 4)) / (len(bg_c) - 1), maps_mul
      with ds.dataset.skip_imread():
        weights = [cnt[v] for _,v,_ in ds]
      for i, w in enumerate(weights):
        with ds.dataset.skip_imread():
          _,l,f = ds[i]
        if l == 0:
          bg = f.split("/")[1]
          weights[i] = 1 / bg_c[bg] * (bg_mul[int(bg == "maps")] / len(bg_c))
      sampler = WeightedRandomSampler(weights, len(ds))
      print("exemplary sampling:")
      with ds.dataset.skip_imread():
        print(Counter(CLASSES[ds[s][1].item()] for s in sampler).most_common())
        print(Counter(ds[s][2].split("/")[1] for s in sampler if ds[s][1] == 0).most_common())
      opts["sampler"] = sampler
    return DataLoader(ds, **opts)


def splitset(ds):
  return random_split(ds, (.8, .2), torch.Generator().manual_seed(42))


def counts(ds, transl=False):
  with ds.dataset.skip_imread():
    c = Counter(l.item() for _,l,_ in ds)
    c.update({i: 0 for i in range(len(CLASSES))})  # there could be missing classes
    if transl:
      return {CLASSES[v]: n for v,n in c.most_common()}
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
    clazzes = CLASSES
  ds = MultiSet(clazzes)
  print(counts(ds, transl=True))
  BS = 256
  dl = MultiSet.load(ds, clazzes, weighted=True,
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

