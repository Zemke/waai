#!/usr/bin/env python3

import sys
import torchvision.transforms as T
# TODO err circular import
from dataset import H, W, CLASSES, MAPS


K = {
  "water": [
    T.RandomResizedCrop((H, W)),
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "text": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "cloud": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "girder": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "barrel": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "blood": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=180, translate=(.2,.2)),
  ],
  "bg": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "mine": [
    T.RandomAffine(degrees=180, translate=(.2,.2)),
    T.RandomHorizontalFlip(p=.5),
  ],
  "worm": [
    T.RandomAffine(degrees=20, translate=(.2,.2)),
    T.RandomHorizontalFlip(p=.5),
  ],
  "dynamite": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
  ],
  "puffs": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomVerticalFlip(p=.5),
    T.RandomResizedCrop((H, W)),
    T.RandomAffine(degrees=180, translate=(.2,.2)),
  ],
  "sheep": [
    T.RandomHorizontalFlip(p=.5),
    T.RandomAffine(degrees=0, translate=(.2,.2)),
    T.RandomResizedCrop((H, W)),
  ],
}

for k in K.keys():
  if k not in CLASSES: print(k, 'is not in', CLASSES)

M = [T.RandomHorizontalFlip(p=.5),]


def augment(clazz):
  if clazz in MAPS:
    return M
  if clazz in K:
    return K[clazz]
  print("no augmentation for", clazz, file=sys.stderr)
  return []


# 169 water
# 200 text
# 279 cloud
# 284 girder
# 332 barrel
# 422 blood
# 427 bg
# 639 mine
# 1337 worm
# 1381 music
# 1631 dynamite
# 1687 puffs
# 1877 forest
# 2323 medieval
# 2326 sheep
# 2340 hospital
# 2424 tentacle
# 2454 sports
# 2534 tools
# 2624 gulf
# 2676 fruit
# 2680 construction
# 2701 -hell
# 2743 desert
# 2791 art
# 2865 urban
# 2892 dungeon
# 2936 pirate
# 2967 cheese
# 3002 snow
# 3007 tribal
# 3009 manhattan
# 3046 time
# 3097 easter
# 3314 hell
# 3339 -beach
# 3532 -farm
# 3662 space
# 3800 -forest
# 3995 -desert
# 4064 jungle
# 93808 in total

