#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2


def load(path):
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def tile(img):
  kernel, stride = 25, 2
  rr = []
  for xi in range(0, len(img), stride):
    for yi in range(0, len(img[xi]), stride):
      # padding dims < 25 with black pixels
      padd = np.zeros((kernel, kernel ,3), np.uint8)
      targ = img[xi:xi+kernel, yi:yi+kernel, :]
      padd[:targ.shape[0], :targ.shape[1], :] = targ
      rr.append(padd)
  return rr


def top(aa, rr, n=100):
  return [aa[idx] for idx in (-np.array(rr)).argsort()[:n]]


def plt_train(loss, acc):
  plt.figure(figsize=(10,10))
  plt.plot(loss, acc)
  plt.show()

