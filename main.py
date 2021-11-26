#!/usr/bin/env python

import sys

import dynanet
import dynaset
import visual


if __name__ == "__main__":
  p1 = sys.argv[1]
  pt = p1.endswith('.pt') or p1.endswith('.pth')
  if pt:
    if not os.path.is_file(p1):
      raise Exception(f'{p1} is not a file')
    net = dynanet.pretrained(path)
    print(f'using existing model {p1}')
  evalu = len(sys.argv) > 2:
  if evalu
    src_dir = sys.argv[1+pt]
    imgs = []
    with os.scandir(src_dir) as it:
      for entry in it:
        if entry.is_file() and entry.name.lower().endswith('.png'):
          imgs.append(entry.path)
    print(f'reading {len(imgs)} pngs from {src_dir}')
    topn = sys.env['TOPN'] or 10
    target_dir = sys.argv[2+pt]
    if not os.path(target_dir).is_dir():
      raise Exception(f"{target_dir} not found")
    print(f'outputting top {topn} into {target_dir}')
    print('continue? [Y/n]', end=': ')
    cont = input()
    if not (cont.lower().strip() == '' or 'y'):
      print('okay, exiting')
      sys.exit()
  if not pt:
    visual.plt(*net.train(net, dynaset.load()))
    loc = './dynanet.pt'
    print(f'overwrite model to {loc}? [y/N]', end=' ')
    ask = input()
    if ask.lower() == 'y':
      dynanet.save(net, loc)
      print('saved')
  if evalu:
    for i in range(len(imgs)):
      print('processing', i, imgs[i])
      rr = dynanet.eval(net, visual.tile(visual.load(imgs[i])))
      top = visual.top(imgs, rr, topn)

