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


# TODO return "sigmoided" from mutlinet
def sigmoid(x):
  return 1/(1+math.e**(-x))


class Runner:
  def __init__(self):
    self.epochs = multinet.EPOCHS
  
  def pretrained(self, path):
    self.net = multinet.pretrained(path)
    print(f'using existing model {path}')
    return self.net

  def dataset(self):
    self.ds = dataset.MultiSet(weapon=os.getenv("WEAPON", None))
    print(f"mean:{self.ds.mean}, std:{self.ds.std}")
    print(f"training on classes: {self.ds.classes}")
    return self.ds

  def net(self):
    self.net = multinet.MultiNet(len(self.ds.classes)).device()
    return self.net

  def train(self, dl_train, dl_test):
    args = self.net, dl_train, dl_test
    return multinet.train(*args)

  def save(self, loc):
    args = self.net, loc
    return multinet.save(*args)
    
  def pred(self):
    y = multinet.pred(net, dataset.transform(fromstdin))
    argmax = np.argmax(y)
    print(y, argmax)
    return dataset.CLASSES[argmax], sigmoid(y[argmax])


  def pred_capture(self, paths, target_dir, topn):
    for i in trange(len(paths)):
      dl = dataset.load(
          dataset.CaptureMultiSet(
              paths[i]), batch_size=800*640, shuffle=False)
      preds = multinet.pred_capture(self.net, dl)
      argsorted = preds.argsort(axis=0)
      for ci in range(1, len(dataset.CLASSES)):
        for toparg in argsorted[:,ci][::-1][:topn]:
          score = round(sigmoid(preds[toparg][ci]) * 10000)
          ts = round(time() * 1000000)
          visual.write_img(
              dl.dataset.tiles[toparg], target_dir,
              f"{dataset.CLASSES[ci]}_{score}_{ts}.png")


if __name__ == "__main__":
  runner = Runner()

  fromstdin = None
  if select.select([sys.stdin, ], [], [], 0.0)[0]:
    # data from stdin
    if len(sys.argv) > 2:
      raise Exception("cannot combine stdin and source dir")
    fromstdin = visual.loadstdin(sys.stdin)
    #if fromstdin.shape != (25,25,3):
    #  raise Exception("shape's gotta be 25,25,3")
  pt = False
  if len(sys.argv) > 1:
    p1 = sys.argv[1]
    pt = p1.endswith('.pt') or p1.endswith('.pth')
    if pt:
      if not os.path.isfile(p1):
        raise Exception(f'{p1} is not a file')
      net = runner.pretrained(os.path.join(p1))
  pred = len(sys.argv) > 2
  if pred:
    src_dir = sys.argv[1+pt]
    imgs = []
    with os.scandir(src_dir) as it:
      for entry in it:
        if entry.is_file() and entry.name.lower().endswith('.png'):
          imgs.append(entry.path)
    print(f'reading {len(imgs)} pngs from {src_dir}')
    topn = os.environ.get('TOPN')
    target_dir = sys.argv[2+pt]
    if not os.path.isdir(target_dir):
      raise Exception(f"{target_dir} not found")
    if topn is None or "." not in topn:
      topn = 10 if topn is None else int(topn)
      print(f'outputting top {topn} matches into {target_dir}')
    else:
      topn = float(topn)
      print(f'outputting probability better than {topn} into {target_dir}')
    print('continue? [Y/n]', end=': ')
    cont = input().lower().strip()
    if cont != 'y' and cont != '':
      print('okay, exiting')
      sys.exit()
  if not pt:
    data = runner.dataset()
    net = runner.net()
    env_test = os.environ.get('TEST')
    if env_test == '1':
      print('test with split set')
      ds_train, ds_test = dataset.splitset(data)
      dl_train, dl_test = \
        dataset.load(ds_train, batch_size=16, oversample=True), \
        dataset.load(ds_test, batch_size=len(ds_test))
      # TODO log value counts from dataset
    else:
      dl_train, dl_test = dataset.load(data, oversample=True), None
    trainres, validres, pcres = runner.train(dl_train, dl_test)
    if os.environ.get("TRAINONLY") != '1':
      visual.plt_res(trainres, validres, pcres, data.classes, runner.epochs)
    if env_test:
      print("saving model with test flag is not possible as it was \
trained with only part of the dataset")
    else:
      loc = "./model_multinet.pt"
      print(f'overwrite model to {loc}? [Y/n]', end=' ')
      if input().strip().lower() != 'n':
        runner.save(loc)
        print('saved')
  if pred:
    print('evaluating')
    runner.pred_capture(imgs, target_dir, topn)
  if fromstdin is not None:
    print(runner.pred())

