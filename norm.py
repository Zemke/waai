#!/usr/bin/env python3

import sys
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'cifar10':
    from torchvision.datasets import CIFAR10
    t = T.Compose([T.ToTensor()])
    ds = CIFAR10('./cifar10', transform=t, download=True)
    dl = DataLoader(ds, batch_size=len(ds), shuffle=True)
    assert len(dl) == 1
    # from the internet I know the mean and std of CIFAR10 is:
    # std=[0.247 0.243 0.261] mean=[0.491 0.482 0.446]
    print(torch.std_mean(next(iter(dl))[0], (0,2,3)))
    exit()

  import dataset
  ds = dataset.splitset(dataset.MultiSet(classes := dataset.classes(os.getenv("WEAPON", None))))[0]
  with ds.dataset.skip_normalize():
    dl = dataset.load(ds, classes, batch_size=len(ds), weighted=True)
    assert len(dl) == 1
    d = next(iter(dl))[0]
    assert d.shape == (len(ds), 3, dataset.H, dataset.W)

    def norm(dd):
      return torch.std_mean(torch.stack([Normalize(std=std, mean=mean)(d) for d in dd]), (0,2,3))

    print('shape of data X, C, H, W:', d.shape)
    std, mean = torch.std_mean(d, (0,2,3))
    print()
    print("std, mean of data")
    print("max precision:", std, mean)
    print("    formatted:", *tuple(tuple(round(y, 3) for y in x.round(decimals=3).tolist()) for x in [std, mean]))
    print()
    print("data is weighted and online augmented therefore differs with each epoch")
    print("applied on same data:", norm(d))
    print(" applied on new data:", norm(next(iter(dl))[0]))

