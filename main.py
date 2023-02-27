#!/usr/bin/env python

import sys
import os
import select
from time import time
import math

from tqdm import trange

import multinet
import dataset
import visual

import numpy as np


class Runner:
  weapon = os.getenv("WEAPON", None)

  def __init__(self):
    self.epochs = multinet.EPOCHS
    self.classes = dataset.classes(self.weapon)
    print("classes:", self.weapon, len(self.classes), self.classes)
  
  def pretrained(self, path):
    self.net = multinet.pretrained(path).device()
    print(f'using existing model {path}')
    if self.net.num_classes != len(self.classes):
      raise Exception(f"{self.net.num_classes} classes in model but {len(self.classes)} expected")

  def net(self):
    self.net = multinet.MultiNet(len(self.classes)).device()

  def dataset(self):
    self.ds = dataset.MultiSet(classes=self.classes)
    print(f"mean:{self.ds.mean}, std:{self.ds.std}")
    return self.ds

  def train(self, dl_train, dl_test):
    args = self.net, dl_train, dl_test
    return multinet.train(*args)
    
  def pred(self, x):
    y = multinet.pred(net, dataset.transform(x))
    argmax = np.argmax(y)
    # TODO output per class probability
    return self.classes[argmax], y[argmax]

  def pred_capture(self, paths, target_dir):
    thres = float(os.getenv('THRES', .8))
    print(f'output threshold >={thres*100}% per tile')

    for i in (progr := trange(len(paths))):
      f = paths[i].split('/')[-1][:-4]
      progr.set_postfix(f=f)
      dl = dataset.load(
        ds := dataset.CaptureMultiSet(paths[i]),
        batch_size=len(ds),
        shuffle=False)
      preds = multinet.pred_capture(self.net, dl)
      mx = preds.argmax(1)
      for i in range(len(ds.tiles)):
        prob = preds[i][mx[i]]
        if prob < thres:
          continue
        visual.write_img(
          ds.tiles[i],
          target_dir,
          f"{self.classes[mx[i]]}_{prob*100:.0f}_{f}_{time()*1e+6:.0f}.png")
      exit()

  def help(self):
    print('                   predict single image: main.py model.pt image.png')
    print('         mass inference from source dir: main.py model.pt source/')
    print('tile captures and inference into target: main.py model.pt source/ target/')
    print('                      train a new model: main.py')


if __name__ == "__main__":
  runner = Runner()
  if len(sys.argv) == 2 and sys.argv[1].startswith('-h'):
    runner.help()
    exit()

  # pretrained model
  if len(sys.argv) > 2:
    if sys.argv[1].endswith('.pt') or sys.argv[1].endswith('.pth'):
      runner.pretrained(path := os.path.join(sys.argv[1]))
      print('using pretrained model', path)
    else:
      print(
        "first param neeeds to be a PyTorch saved model ending in .pt or .pth",
        file=sys.stderr)
      exit(2)

    if len(sys.argv) == 3:
      src = os.path.join(sys.argv[2])
      if os.path.isdir(src):
        # mass predict images
        print('mass predicting images is not yet supported.', file=sys.stderr)
        exit(3)
      elif os.path.isfile(src):
        # predict single image
        print('predicting single image', src)
        print(runner.pred(visual.load(src)))
        exit()
    elif len(sys.argv) == 4:
      # predicting tiled captures
      src_dir, target_dir = sys.argv[2], sys.argv[3]
      imgs = []
      with os.scandir(src_dir) as it:
        for entry in it:
          if entry.is_file() and entry.name.lower().endswith('.png'):
            imgs.append(entry.path)
      if not os.path.isdir(target_dir):
        raise Exception(f"{target_dir} not found")
      print(runner.pred_capture(imgs, target_dir))
      exit()

  # train new model
  elif len(sys.argv) == 1:
    data = runner.dataset()
    runner.net()
    print('dataset:', dataset.counts(data.classes, data, transl=True))
    ds_train, ds_test = dataset.splitset(data)
    print('test seed:', x := dataset.counts(runner.classes, ds_test, transl=True))
    print('train seed:', y := dataset.counts(runner.classes, ds_train, transl=True))
    dl_train, dl_test = \
      dataset.load(ds_train, runner.classes, weighted=True), \
      dataset.load(ds_test, runner.classes, weighted=True, batch_size=len(ds_test))
    trainres, testres, pcres = runner.train(dl_train, dl_test)
    visual.plt_res(trainres, testres, pcres, data.classes, runner.epochs)

  else:
    print("invalid number of parameters", file=sys.stderr)
    runner.help()
    exit(10)

