#!/usr/bin/env python

import os 
import json
from PIL import Image

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

import dataset


class CropSet(Dataset):
  def __init__(self):
    self.files = []
    self.transform = T.Compose([T.ToTensor()])
    with os.scandir("./labelling/target") as it:
      for entry in it:
        if entry.is_file() and entry.name.lower().endswith('.json'):
          self.files.append(entry.path)

  def augment(self):
    class AugmentSet(Dataset):
      def __init__(self, default_transforms):
        self.files = []
        self.def_transf = default_transforms
      def add(self, f, aug_transf, crop_params=None):
        self.files.append((f, aug_transf, crop_params))
      def __len__(self):
        return len(self.files)
      def __getitem__(self, idx):
        return CropSet._getitem(
          self.files[idx][0],
          T.Compose([self.files[idx][1], *self.def_transf.transforms]),
          self.files[idx][2])

    ds = AugmentSet(self.transform)
    for f in self.files:
      if f.split('/')[-1].startswith('hd_'):
        crop_params = T.RandomCrop.get_params(torch.zeros(1080,1920), (800,1000))  # i,j,th,tw
        ds.add(f, lambda img, crop_params=crop_params: F.crop(img, *crop_params), crop_params)
      ds.add(f, T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 3)))
    return self + ds

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    return self._getitem(self.files[idx], self.transform)

  @staticmethod
  def _getitem(file, transform, crop_params=None):
    with open(file, 'r') as f:
      j = json.loads(f.read())
    flip = crop_params is None and torch.rand(1) < .3
    for doc in j:
      img = Image.open(os.path.join("captures", doc["image"])).convert('RGB')
      transed = F.hflip(transform(img)) if flip else transform(img)
      h,w = transed.shape[1:]
      boxes, labels = [], []
      for annot in doc["annotations"]:
        coord = annot["coordinates"]
        x, y, width, height = coord["x"], coord["y"], coord["width"], coord["height"]
        x1, y1, x2, y2 = x-width/2, y-height/2, x+width/2, y+height/2 
        box = None
        if crop_params is None:
          box = [x1, y1, x2, y2]
        else:
          crop_i,crop_j,*_ = crop_params
          box = [x1-crop_j, y1-crop_i, x2-crop_j, y2-crop_i]
          if box[2] <= 0 or box[0] >= w or box[3] <= 0 or box[1] >= h \
              or box[2]-box[0] < 11 or box[3]-box[1] < 11:
            continue
          box = [max(0,box[0]), max(0,box[1]), min(w,box[2]), min(h,box[3])]
        if flip:
          fliplist = list(range(w+1))[::-1]
          box[0] = fliplist[int(box[0]+width)]
          box[2] = fliplist[int(box[2]-width)]
        boxes.append(box)
        labels.append(dataset.CLASSES.index(annot["label"]))
    # show with bounding boxes
    #from torchvision.utils import draw_bounding_boxes
    #bb = draw_bounding_boxes(
    #  (transed*255).to(torch.uint8),
    #  torch.as_tensor(boxes, dtype=torch.float),
    #  [dataset.CLASSES[label] for label in  labels])
    #import matplotlib.pyplot as plt
    #assert len(labels) == len(boxes)
    #plt.title(file  + " - " + str(crop_params is not None) + " - " + str(len(boxes)))
    #plt.imshow(bb.permute((1,2,0)))
    #plt.show()
    return transed, {
      "boxes": torch.as_tensor(boxes, dtype=torch.float),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


def collate_fn(batch):
  return tuple(zip(*batch))


def load(dataset, batch_size=4, shuffle=True):
  return DataLoader(
      dataset, num_workers=2, persistent_workers=True,
      batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

