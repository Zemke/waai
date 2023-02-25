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
  
  def pretrained(self, path):
    self.net = multinet.pretrained(path).device()
    print(f'using existing model {path}')
    return self.net

  def dataset(self):
    self.ds = dataset.MultiSet(weapon=self.weapon)
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
    
  def pred(self, x):
    y = multinet.pred(net, dataset.transform(x))
    argmax = np.argmax(y)
    # TODO output per class probability
    # TODO what are the classes for the pretrained model?
    return dataset.CLASSES[argmax], y[argmax]

  def pred_capture(self, paths, target_dir):
    thres = float(os.getenv('THRES', .8))
    print(f'output threshold >={thres*100}% per tile')

    for i in (progr := trange(len(paths))):
      f = paths[i].split('/')[-1][:-4]
      progr.set_postfix(f=f)
      dl = dataset.load(
        ds := dataset.CaptureMultiSet(paths[i]),
        # TODO what are the classes for the pretrained model?
        classes := ['-beach', '-desert', '-farm', '-forest', '-hell', 'art', 'barrel', 'blood', 'cheese', 'cloud', 'construction', 'desert', 'dungeon', 'dynamite', 'easter', 'flag', 'forest', 'fruit', 'girder', 'gulf', 'healthbar', 'hell', 'hospital', 'jungle', 'manhattan', 'medieval', 'mine', 'music', 'phone', 'pirate', 'puffs', 'sheep', 'snow', 'space', 'sports', 'tentacle', 'text', 'time', 'tools', 'tribal', 'urban', 'water', 'wind', 'worm'],
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
          f"{classes[mx[i]]}_{prob*100:.0f}_{f}_{time()*1e+6:.0f}.png")
      exit()


if __name__ == "__main__":
  runner = Runner()

  fromstdin = None
  if select.select([sys.stdin, ], [], [], 0.0)[0]:
    # data from stdin
    if len(sys.argv) > 2:
      raise Exception("cannot combine stdin and source dir")
    fromstdin = visual.loadstdin(sys.stdin)
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
    target_dir = sys.argv[2+pt]
    if not os.path.isdir(target_dir):
      raise Exception(f"{target_dir} not found")
  if not pt:
    data = runner.dataset()
    net = runner.net()
    env_test = os.environ.get('TEST')
    print('dataset:')
    print(dataset.counts(data.classes, data, transl=True))
    if env_test == '1':
      print('test with split set')
      ds_train, ds_test = dataset.splitset(data)
      dl_train, dl_test = \
        dataset.load(ds_train, data.classes, weighted=True), \
        dataset.load(ds_test, data.classes, weighted=True, batch_size=len(ds_test))
      print('valid seed:')
      print(dataset.counts(data.classes, ds_test, transl=True))
      print('train seed:')
      print(dataset.counts(data.classes, ds_train, transl=True))
    else:
      dl_train, dl_test = dataset.load(data, data.classes, weighted=True), None
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
    runner.pred_capture(imgs, target_dir)
  if fromstdin is not None:
    print(runner.pred(fromstdin))

