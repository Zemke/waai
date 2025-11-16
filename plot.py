#!/usr/bin/env python3

import os
import re
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w

ion = os.getenv("ION", 1) != '0'
if ion:
  plt.ion()
plt.style.use('dark_background')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())


y = deque([], int(os.getenv("SIZE", 1_000)))
G = L = None
read = int(os.getenv("READ", 0))
while True:
  y_bef = len(y)
  with open('train.log', 'r') as f:
    buff = ''
    for i,line in enumerate(f):
      if i < read:
        continue
      read += 1
      buff += line.replace("\n",'')
      fa = re.findall('{.+?}', buff)
      if len(fa) > 0:
        for fa1 in fa:
          if 'classifier' not in fa1:
            continue
          l = re.sub("device.+?',", '', fa1)
          print('plt', l)
          if read % int(os.getenv("STEP", 2)) == 0:
            y.append(l)
        buff = ''
  if y_bef == len(y):
    continue
  if len(y) < 1:
    continue
  z = [[float(n) for n in re.sub("[^0-9.]+", ",", y1).replace(",0,", ",")[1:-1].split(",")] for y1 in y]
  zt = np.transpose(z)
  labels = [f"{v} {k}" for k,v in zip(re.findall("'(.*?)'", y[-1]), z[-1])]
  if G is None:
    G = [None] * len(labels)
  colors = ['magenta', 'yellow', 'cyan', 'lime', 'red', 'orange', 'blue', 'green'][:len(labels)]
  for i,l in enumerate(labels):
    if G[i] is not None:
      G[i][0].remove()
    G[i] = plt.plot(moving_average(zt[i], max([int(len(y)*(os.getenv("MA", 1/40))), 1])), label=l, color=colors[i])
  if L is not None:
    L.remove()
  L = plt.legend([g for g, in G], labels)
  if ion:
    plt.pause(2)
  else:
    plt.show()
    break

