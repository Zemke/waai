#!/usr/bin/env python

import os
import sys
import pickle
from time import time

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image


def load(path):
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def loadstdin(stdin):
  buff = np.frombuffer(stdin.buffer.read(), dtype='uint8')
  return cv2.cvtColor(cv2.imdecode(buff, 1), cv2.COLOR_BGR2RGB)


def tile(img, kernel=25, stride=2):
  res = []
  for xi in range(0, len(img), stride):
    for yi in range(0, len(img[xi]), stride):
      # padding dims < 25 with black pixels
      padd = np.zeros((kernel, kernel ,3), np.uint8)
      targ = img[xi:xi+kernel, yi:yi+kernel, :]
      padd[:targ.shape[0], :targ.shape[1], :] = targ
      res.append(padd)
  return res


def plt_res(trainres, testres, pcres, classes, epochs):
  with open(f'metrics_{int(time()*1000000)}.pkl', 'wb') as f:
    pickle.dump(
      dict(
        trainres=trainres,
        testres=testres,
        pcres=pcres,
        classes=classes,
        epochs=epochs),
      f)

  trainloss, trainacc = trainres
  testloss, testacc = testres
  pcloss, pcacc = [np.concatenate(x).reshape((-1,len(classes))).transpose((1,0)) for x in pcres]

  plt.style.use("dark_background")
  plt.rcParams["figure.figsize"] = (15,10)
  plt.rcParams["savefig.dpi"] = 200
  fig, ((lossax, accax), (losspcax, accpcax)) = plt.subplots(2, 2)

  lsa = np.linspace(1, epochs, len(trainloss))
  lsb = np.linspace(1, epochs, pcloss.shape[1])

  lossax.plot(lsa, trainloss, label='train')
  lossax.plot(lsa, testloss, label='test')
  lossax.set_ylabel('mean loss')

  accax.plot(lsa, trainacc, label='train')
  accax.plot(lsa, testacc, label='test')
  accax.set_ylabel('mean accuracy')

  pclosslines = []
  colors = {}
  for i in (-pcloss[:,-1]).argsort():
    x, = losspcax.plot(lsb, pcloss[i,:], label=f"{classes[i]} {pcloss[i,-1]:.4f}")
    colors[classes[i]] = x.get_color()
    pclosslines.append(x)
  losspcax.set_ylabel('per-class mean loss')

  pcacclines = []
  for i in (-pcacc[:,-1]).argsort():
    x, = accpcax.plot(
      lsb,
      pcacc[i,:],
      color=colors[classes[i]],
      label=f"{classes[i]} {round(pcacc[i,-1]*100)}%")
    pcacclines.append(x)
  accpcax.set_ylabel('per-class mean accuracy')

  losspcax.set_xlabel('epoch')
  accpcax.set_xlabel('epoch')

  lossax.legend()
  accax.legend()
  losspcleg = losspcax.legend(fontsize=5.5, labelspacing=0)
  accpcleg = accpcax.legend(fontsize=5.5, labelspacing=0)

  lined = [{}, {}]
  for legline, origline in zip(losspcleg.get_lines(), pclosslines):
    legline.set_picker(True)
    legline.set_linewidth(4.0)
    lined[0][legline] = origline
  for legline, origline in zip(accpcleg.get_lines(), pcacclines):
    legline.set_picker(True)
    legline.set_linewidth(4.0)
    lined[1][legline] = origline

  def on_pick(event):
    li = lined[::-1] if event.artist.axes == accpcax else lined
    if all(v.get_visible() for v in li[0].values()):
      # all are visible
      for k,v in li[0].items():
        if k != event.artist:
          v.set_visible(False)
          k.set_alpha(.2)
    elif len(vis := [(k,v) for k,v in li[0].items() if v.get_visible()]) == 1 \
      and vis[0][0] == event.artist:
      # one is visible and it's clicked
      for k,v in li[0].items():
        v.set_visible(True)
        k.set_alpha(1.)
    else:
      legline = event.artist
      origline = li[0][legline]
      visible = not origline.get_visible()
      origline.set_visible(visible)
      legline.set_alpha(1. if visible else .2)
    # cascade to accpc
    for v in li[0].values():
      name, visible = v.get_label().split(' ')[0], v.get_visible()
      for k1, v1 in li[1].items():
        if v1.get_label().split(' ')[0] == name:
          v1.set_visible(visible)
          k1.set_alpha(1. if visible else .2)
    fig.canvas.draw()
  fig.canvas.mpl_connect('pick_event', on_pick)

  plt.subplots_adjust(
    left=.05, right=.99, top=.99, bottom=0.05,
    wspace=.12, hspace=.1)
  plt.show()


def write_img(img, path, name):
  return Image.fromarray(img).save(os.path.join(path, name))


def deserialize(p='metrics.pkl'):
  with open(p, 'rb') as f:
    X = pickle.load(f)
  plt_res(**X)


if __name__ == "__main__":
  deserialize(*sys.argv[1:2])

