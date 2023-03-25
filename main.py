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


def _worker(q, thres, target_dir):
  while 1:
    if (deq := q.get()) == "DONE":
      break
    tiles, origs, preds, f = deq
    mx = preds.argmax(1)
    for k in range(len(tiles)):
      prob = preds[k][mx[k]]
      if prob < thres:
        continue
      clazz = dataset.CLASSES[mx[k]]
      visual.write_img(
        origs[k],
        os.path.join(target_dir, clazz),
        f"{clazz}_{prob*100:.0f}_{f}_{time()*1e+6:.0f}.png")


class Runner:
  
  def pretrained(self, path):
    self.net = multinet.pretrained(path)
    print(f'using existing model {path}')
    if self.net.num_classes != len(dataset.CLASSES):
      raise Exception(f"{self.net.num_classes} classes in model but {len(dataset.CLASSES)} expected")

  def net(self):
    self.net = multinet.MultiNet(len(dataset.CLASSES))

  def dataset(self):
    self.ds = dataset.MultiSet()
    return self.ds

  def train(self, dl_train, ds_test):
    trainer = multinet.Trainer(
      self.net,
      dataset.CLASSES,
      dl_train,
      multinet.Tester(self.net, ds_test, dataset.counts(ds_test)))
    return tuple([*trainer(), trainer.epochs])
    
  def pred(self, x):
    y = multinet.pred(self.net, dataset.transform(x))
    return sorted({(dataset.CLASSES[i], round(y1.item()*100)) for i, y1 in enumerate(y)}, key=lambda v: v[1])

  def pred_capture(self, source_dir, target_dir):
    thres = float(os.getenv('THRES', .8))
    print(f'output threshold >={thres*100}% per tile')

    for c in dataset.CLASSES:
      os.mkdir(os.path.join(target_dir, c))

    from multiprocessing import Process, Queue
    (p := Process(target=_worker, args=(q := Queue(), thres, target_dir))).start()

    dl = dataset.CaptureSet.load(ds := dataset.CaptureSet(source_dir, target_dir))
    for i, (tiles, origs) in (progr := tqdm(enumerate(dl), total=len(dl))):
      f = ds.imgs[i].split('/')[-1][:-4]
      progr.set_postfix(f=f)
      q.put((tiles, origs, multinet.pred_capture(self.net, tiles), f))
    q.put("DONE")
    q.join()

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
  print('classes', dataset.CLASSES)

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
    runner.net()
    ds_train, ds_test = dataset.splitset(data)
    if int(os.getenv('DEBUG', 0)) > 0:
      print('all data', dataset.counts(data, transl=True))
      print('train data', dataset.counts(ds_train, transl=True))
      print('test data', dataset.counts(ds_test, transl=True))
    dl_train = dataset.MultiSet.load(ds_train, weighted=True)
    plots, conf_mat, epochs = runner.train(dl_train, ds_test)
    visual.plt_res(**plots, conf_mat=conf_mat, epochs=epochs, classes=dataset.CLASSES)
    exit()

  else:
    print("invalid number of parameters", file=sys.stderr)
    runner.help()
    exit(10)

