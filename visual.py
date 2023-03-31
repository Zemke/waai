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


def plt_res(loss, acc, test_loss, test_acc, test_acc_pc, conf_mat, epochs, classes, save=True):

  if save:
    args = locals()
    del args["save"]
    if "self" in args:
      del args["self"]
    with open(f'metrics_{int(time()*1000000)}.pkl', 'wb') as f:
      pickle.dump(args, f)

  plt.style.use("dark_background")
  plt.rcParams["figure.figsize"] = (10,8)
  plt.rcParams["savefig.dpi"] = 200
  fig, ((lossax, accax), (accpcax, mat)) = plt.subplots(2, 2)

  lossax.set_ylabel('mean loss')
  accax.set_ylabel('mean accuracy')
  accpcax.set_ylabel('per-class mean accuracy')

  lossax.set_xlabel('epoch')
  accax.set_xlabel('epoch')
  accpcax.set_xlabel('epoch')

  linsp = np.linspace(1, epochs, len(loss))
  if not (len(test_loss) == len(test_acc) == len(test_acc_pc)):
    raise Exception("test metrics must all have the same length")
  linsp_test = np.linspace(1, epochs, len(test_loss))

  lossax.plot(linsp, loss, label='train')
  lossax.plot(linsp_test, test_loss, label='test')

  accax.plot(linsp, acc, label='train')
  accax.plot(linsp_test, test_acc, label='test')

  pcacc = np.array(test_acc_pc).transpose((1,0))
  pcacclines = []
  colors = {}  # remember colors for potential future use
  for i in (-pcacc[:,-1]).argsort():
    x, = accpcax.plot(
      linsp_test,
      pcacc[i,:],
      label=f"{classes[i]} {round(pcacc[i,-1]*100)}%")
    colors[classes[i]] = x.get_color()
    pcacclines.append(x)

  lossax.legend()
  accax.legend()
  accpcleg = accpcax.legend(fontsize=5.5, labelspacing=0)

  lined = {}
  for legline, origline in zip(accpcleg.get_lines(), pcacclines):
    legline.set_picker(True)
    legline.set_linewidth(4.0)
    lined[legline] = origline

  def on_pick(event):
    if all(v.get_visible() for v in lined.values()):
      # all are visible
      for k,v in lined.items():
        if k != event.artist:
          v.set_visible(False)
          k.set_alpha(.2)
    elif len(vis := [(k,v) for k,v in lined.items() if v.get_visible()]) == 1 \
      and vis[0][0] == event.artist:
      # one is visible and it's clicked
      for k,v in lined.items():
        v.set_visible(True)
        k.set_alpha(1.)
    else:
      legline = event.artist
      origline = lined[legline]
      visible = not origline.get_visible()
      origline.set_visible(visible)
      legline.set_alpha(1. if visible else .2)
    fig.canvas.draw()
  fig.canvas.mpl_connect('pick_event', on_pick)

  weights = (recip := 1 / conf_mat.sum(0)) / sum(recip)
  weighted_conf_mat = ((conf_mat * weights.repeat(7,1)) * 10).round().int()
  mat.matshow(weighted_conf_mat, cmap='seismic')
  mat.set_xticks(range(len(classes)), classes, size='small')
  mat.set_yticks(range(len(classes)), classes, size='small')
  mat.set_ylabel("prediction")
  mat.set_xlabel("label")
  for (i, j), z in np.ndenumerate(weighted_conf_mat):
    mat.text(j, i, str(z), ha='center', va='center')
  mat.text(
    len(classes) // 2, len(classes) + 1,
    'weighted confusion matrix at min test loss',
    size='small', ha='center')

  plt.tight_layout()
  plt.show()


def write_img(img, path, name):
  return Image.fromarray(img).save(os.path.join(path, name))


def deserialize(p='metrics.pkl'):
  with open(p, 'rb') as f:
    X = pickle.load(f)
  plt_res(**X, save=False)


if __name__ == "__main__":
  deserialize(*sys.argv[1:2])

