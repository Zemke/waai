#!/usr/bin/env python

import sys
import os
from time import time

from tqdm import trange

import dynanet
import dynaset
import visual


if __name__ == "__main__":
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
    visual.plt_train(*dynanet.train(net, dynaset.load()))
    loc = './dynanet.pt'
    print(f'overwrite model to {loc}? [y/N]', end=' ')
    ask = input().strip().lower()
    if ask == 'y':
      dynanet.save(net, loc)
      print('saved')
  if evalu:
    print('evaluating')
    for i in trange(len(imgs)):
      tiles = visual.tile(visual.load(imgs[i]))
      rr = dynanet.eval(net, tiles)
      #import numpy as np
      #np.save("rr.npy", rr)
      #with open('./rr.npy', 'rb') as f:
      #  rr = np.load(f)
      if isinstance(topn, int):
        top, topargs = visual.topk(tiles, rr, topn)
      else:
        top, topargs = visual.topprob(tiles, rr, topn)
      for i in range(len(top)):
        ts = round(time() * 1000000)
        score = round(rr[topargs[i]]*100)
        visual.write_img(top[i], target_dir, f"{score}_{ts}.png")

