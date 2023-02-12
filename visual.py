#!/usr/bin/env python

import os
import sys
import pickle

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


def topk(aa, rr, k=100):
  sargs = np.argsort(rr)[-k:][::-1]
  return [aa[i] for i in sargs], sargs


def topprob(aa, rr, prob=.5):
  res = []
  sargs = (-rr).argsort()
  for i in range(len(sargs)):
    if rr[sargs[i]] > prob:
      res.append(aa[sargs[i]])
    else:
      break
  return res, sargs[:len(res)]


def plt_res(trainres, validres, pcres, classes, epochs):
  with open('metrics.pkl', 'wb') as f:
    pickle.dump(
      dict(
        trainres=trainres,
        validres=validres,
        pcres=pcres,
        classes=classes,
        epochs=epochs),
      f)

  trainloss, trainacc = trainres
  validloss, validacc = validres
  pcloss, pcacc = [np.concatenate(x).reshape((-1,len(classes))).transpose((1,0)) for x in pcres]

  plt.style.use("dark_background")
  plt.rcParams["figure.figsize"] = (15,10)
  plt.rcParams["savefig.dpi"] = 200
  fig, ((lossax, accax), (losspcax, accpcax)) = plt.subplots(2, 2)

  lsa = np.linspace(1, epochs, len(trainloss))
  lsb = np.linspace(1, epochs, pcloss.shape[1])

  lossax.plot(lsa, trainloss, label='train')
  lossax.plot(lsa, validloss, label='valid')
  lossax.set_ylabel('mean loss')

  accax.plot(lsa, trainacc, label='train')
  accax.plot(lsa, validacc, label='valid')
  accax.set_ylabel('mean accuracy')

  for i in (-pcloss[:,-1]).argsort():
    losspcax.plot(lsb, pcloss[i,:], label=f"{classes[i]} {pcloss[i,-1]:.4f}")
  losspcax.set_ylabel('per-class mean loss')

  for i in (-pcacc[:,-1]).argsort():
    accpcax.plot(lsb, pcacc[i,:], label=f"{classes[i]} {round(pcacc[i,-1]*100)}%")
  accpcax.set_ylabel('per-class mean accuracy')

  losspcax.set_xlabel('epoch')
  accpcax.set_xlabel('epoch')

  lossax.legend()
  accax.legend()
  losspcax.legend(fontsize=5.5, labelspacing=0)
  accpcax.legend(fontsize=5.5, labelspacing=0)

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

