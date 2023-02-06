#!/usr/bin/env python

import os
import sys
import cv2

import visual
from tqdm import tqdm

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print("python3 tile.py input_dir output_dir")
    exit(1)
  DIR = sys.argv[1]
  OUT = sys.argv[2]

  for f in tqdm(os.listdir(DIR)):
    if not f.lower().endswith('.png'):
      continue
    img = cv2.imread(os.path.join(DIR, f))
    tiles = visual.tile(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), kernel=30, stride=30)
    for i, tile in enumerate(tiles):
      if cv2.countNonZero(cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)) >= 400:
        visual.write_img(tile, OUT, f"{f[:-4]}_{i}.png")

