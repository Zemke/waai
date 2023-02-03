#!/usr/bin/env python

import os
import sys
import cv2

import visual

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print("python3 tile.py input_dir output_dir")
    exit(1)
  DIR = sys.argv[1]
  OUT = sys.argv[2]

  for f in os.listdir(DIR):
    if not f.lower().endswith('.png'):
      continue
    img = visual.load(os.path.join(DIR, f))
    tiles = visual.tile(img, kernel=30, stride=15)
    for i, tile in enumerate(tiles):
      if cv2.countNonZero(tile) >= 100:
        visual.write_img(tile, OUT, f"{f[:-4]}_{i}.png")

