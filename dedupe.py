#!/usr/bin/env python

import os
import sys
import time

from PIL import Image
import numpy as np
from natsort import natsorted
from tqdm import trange

THRES = 20

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print("python3 tile.py input_dir output_dir")
    exit(1)
  DIR = sys.argv[1]
  OUT = sys.argv[2]

  I = []
  N = []
  for f in natsorted(os.listdir(DIR)):
    if not f.lower().endswith('.png'):
      continue
    img = Image.open(os.path.join(DIR, f))
    N.append(img.filename.split('/')[-1])
    I.append(img.resize((30,30)).convert("RGB"))

  dels = set()
  for i in trange(0, len(I)-1, leave=False):
    if i in dels:
      continue
    fi = I[i]
    for j in range(i+1, len(I)):
      if j in dels:
        continue
      fj = I[j]
      mae = np.abs(np.subtract(fi, fj)).mean().item()
      if mae <= THRES:
        name = str(round(mae)) + f'_{j-i}_' + str(time.time_ns() // 1000000)
        cat = Image.fromarray(np.concatenate((fi, fj), axis=1))
        cat.save(os.path.join(OUT, name + '.png'), "png")
        print(mae, N[i], N[j])
        dels.add(j)

