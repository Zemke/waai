#!/usr/bin/env python

import sys
import os
import select
from time import time
import math

from tqdm import trange

import dynanet
import dynaset
import visual


def sigmoid(x):
    return 1/(1+math.e**(-x))

if __name__ == "__main__":
  fromstdin = None
  if select.select([sys.stdin, ], [], [], 0.0)[0]:
    # data from stdin
    if len(sys.argv) > 2:
      raise Exception("cannot combine stdin and source dir")
    fromstdin = visual.loadstdin(sys.stdin)
    if fromstdin.shape != (25,25,3):
      raise Exception("shape's gotta be 25,25,3")
  pt = False
  if len(sys.argv) > 1:
    p1 = sys.argv[1]
    pt = p1.endswith('.pt') or p1.endswith('.pth')
    if pt:
      if not os.path.isfile(p1):
        raise Exception(f'{p1} is not a file')
      net = dynanet.pretrained(os.path.join(p1))
      print(f'using existing model {p1}')
  evalu = len(sys.argv) > 2
  if evalu:
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
    net = dynanet.DynaNet()
    data = dynaset.DynaSet()
    env_test = os.environ.get('TEST')
    if env_test == '1':
      print('test with split set')
      ds_train, ds_test = dynaset.splitset(data)
      dl_train, dl_test = dynaset.load(ds_train), dynaset.load(ds_test)
    else:
      dl_train, dl_test = dynaset.load(data), None
    trainres, validres = dynanet.train(net, dl_train, dl_test)
    visual.plt_res(trainres, validres, dynanet.EPOCHS)
    loc = './dynanet.pt'
    if env_test:
      print("saving model with test flag is not possible as it was \
trained with only part of the dataset")
    else:
      print(f'overwrite model to {loc}? [y/N]', end=' ')
      ask = input().strip().lower()
      if ask == 'y':
        dynanet.save(net, loc)
        print('saved')
  if evalu:
    print('evaluating')
    for i in trange(len(imgs)):
      dl = dynaset.load(
          dynaset.CaptureSet(
              imgs[i]), batch_size=512, shuffle=False)
      rr = dynanet.pred_capture(net, dl)
      #import numpy as np
      #np.save("rr.npy", rr)
      #with open('./rr.npy', 'rb') as f:
      #  rr = np.load(f)
      if isinstance(topn, int):
        top, topargs = visual.topk(dl.dataset.tiles, rr, topn)
      else:
        top, topargs = visual.topprob(dl.dataset.tiles, sidmoid(rr), topn)
      for i in range(len(top)):
        ts = round(time() * 1000000)
        score = round(sigmoid(rr[topargs[i]]) * 10000)
        visual.write_img(top[i], target_dir, f"{score}_{ts}.png")
  if fromstdin is not None:
    print(sigmoid(dynanet.pred(net, fromstdin)[0]))  # sigmoid

