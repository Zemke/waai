#!/usr/bin/env python

import os 
import json
from PIL import Image
import sys
import time

with os.scandir("target") as it:
  for entry in it:
    if entry.is_file() and entry.name.lower().endswith('.json'):
      with open(entry.path, 'r') as f:
        j = json.loads(f.read())
      for doc in j:
        img = Image.open("source/" + doc["image"])
        for annot in doc["annotations"]:
          coord = annot["coordinates"]
          x, y, width, height = coord["x"], coord["y"], coord["width"], coord["height"]
          crop = img.crop((x-width/2, y-height/2, x+width/2, y+height/2))
          #crop.show()
          ts = round(time.time() * 10000)
          filen = f"crops/{annot['label']}/{doc['image'][:-4]}_{ts}.png"
          crop.save(filen)

