#!/usr/bin/env python

import sys
import os
import select
from time import time
import math

import multinet
import dataset
import visual

from tqdm import tqdm
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
    return self.ds

  def train(self, dl_train, dl_test):
    args = self.net, dl_train, dl_test
    return multinet.train(*args)
    
  def pred(self, x):
    y = multinet.pred(net, dataset.transform(x))
    argmax = np.argmax(y)
    # TODO output per class probability
    return self.classes[argmax], y[argmax]

  def pred_capture(self, source_dir, target_dir):
    thres = float(os.getenv('THRES', .8))
    print(f'output threshold >={thres*100}% per tile')
    dl = dataset.CaptureSet.load(ds := dataset.CaptureSet(source_dir, target_dir))
    for i, (tiles, origs) in (progr := tqdm(enumerate(dl))):
      f = ds.imgs[i].split('/')[-1][:-4]
      progr.set_postfix(f=f)
      preds = multinet.pred_capture(self.net, tiles)
      mx = preds.argmax(1)
      for k in range(len(tiles)):
        prob = preds[k][mx[k]]
        if prob < thres:
          continue
        visual.write_img(
          origs[k],
          target_dir,
          f"{self.classes[mx[k]]}_{prob*100:.0f}_{f}_{time()*1e+6:.0f}.png")

  @staticmethod
  def help():
    print('                   predict single image: main.py model.pt image.png')
    print('         mass inference from source dir: main.py model.pt source/')
    print('tile captures and inference into target: main.py model.pt source/ target/')
    print('                      train a new model: main.py')


if __name__ == "__main__":
  if len(sys.argv) == 2 and sys.argv[1].startswith('-h'):
    Runner.help()
    exit()
  runner = Runner()

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
      print(runner.pred_capture(sys.argv[2], sys.argv[3]))
      exit()

  # train new model
  elif len(sys.argv) == 1:
    data = runner.dataset()
    print('all data', dataset.counts(data, transl=True))
    runner.net()
    ds_train, ds_test = dataset.splitset(data)
    print('train data', dataset.counts(ds_train, transl=True))
    print('test data', dataset.counts(ds_test, transl=True))
    dl_train = dataset.MultiSet.load(
      ds_train, runner.classes, weighted=True)
    dl_test = dataset.MultiSet.load(
      ds_test, runner.classes, weighted=True, batch_size=len(ds_test))
    trainres, testres, pcres = runner.train(dl_train, dl_test)
    visual.plt_res(trainres, testres, pcres, data.classes, runner.epochs)
    exit()

  else:
    print("invalid number of parameters", file=sys.stderr)
    runner.help()
    exit(10)

