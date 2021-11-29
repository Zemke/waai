#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image


def load(path):
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def loadstdin(stdin):
  buff = np.frombuffer(stdin.buffer.read(), dtype='uint8')
  return cv2.cvtColor(cv2.imdecode(buff, 1), cv2.COLOR_BGR2RGB)

def tile(img):
  kernel, stride = 25, 2
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
  sargs = np.argsort(rr)[::-1]
  for i in range(len(sargs)):
    if rr[sargs[i]] > prob:
      res.append(aa[sargs[i]])
    else:
      break
  return res, sargs[:len(res)]


def plt_res(trainres, validres):
  trainloss, trainacc = trainres
  validloss, validacc = validres

  trainax = plt.subplot(211)
  trainax.set_title('train')
  plt.plot(trainloss, label='loss')
  plt.plot(trainacc, label='acc')

  validax = plt.subplot(212)
  validax.set_title(f'valid')
  meanloss = (np.mean(validloss)*100).round(4)
  meanacc = (np.mean(validacc)*100).round(4)
  plt.plot(validloss, label=f'loss m={meanloss}')
  plt.plot(validacc, label=f'acc m={meanacc}')

  plt.legend()
  plt.show()


def write_img(img, path, name):
  return Image.fromarray(img).save(os.path.join(path, name))

