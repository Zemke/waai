from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from os import listdir
from os.path import isfile, join

for f in [f for f in listdir("weapons") if isfile(join("weapons", f))]:
  img = Image.open('weapons/' + f)
  img = img.convert("RGBA")
  datas = img.getdata()

  newData = []
  for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
      newData.append((0, 0, 0, 0))
    else:
      newData.append(item)

  img.putdata(newData)
  img.save("weapons_alpha/" + f, "PNG")

