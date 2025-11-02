from os import listdir
from os.path import isfile, join
from random import randrange

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import v2
import torchvision.transforms.functional as F

M = ['target/' + f for f in listdir("target") if isfile(join("target", f)) and f.split(".")[-1] == "png"]
W = ['worms/' + f for f in listdir("worms") if isfile(join("worms", f)) and f.split(".")[-1] == "png"]
C = [
  "weapons_alpha/mine.png",
  "weapons_alpha/barrel.png",
  "weapons_alpha/dynamite.png",
]
S = [30, 25, 40, 25]
L = ['worm', 'mine', 'barrel', 'dynamite']

class DynamicSet(Dataset):
  def __init__(self):
    #self.transform = T.Compose([T.ToTensor()])
    pass

  def __len__(self):
    return len(100_000)

  def __getitem__(self, idx):
    im1 = Image.open(M[randrange(len(M))]).convert("RGBA")
    wi = 0
    SP = 45
    height, width, _ = np.array(im1).shape
    print( np.array(im1).shape)
    im1 = im1.convert("RGBA")
    back_im = im1.copy().convert("RGBA")
    boxes, labels = [], []
    for y in range(SP-30, height-SP, SP):
      for x in range(SP-30, width-SP, SP):
        a = Image.new("RGBA", back_im.size, (0,0,0,0))
        c = W[randrange(len(W))] if (t := randrange(4)) == 0 else C[t-1]
        im2 = Image.open(c).convert("RGBA").resize((S[t], S[t]))
        wi += 1
        a.paste(im2, (x1 := x+randrange(10), y1 := y+randrange(10)))
        back_im = Image.alpha_composite(back_im, a)
        labels.append(t)
        boxes.append([x1, y1, x1+S[t], y1+S[t]])
    back_im = back_im.convert("RGB")
    back_im.show()
    return \
      T.Compose([
        T.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        # no norm in rcnn
        #T.Normalize(torch.tensor([0.8580, 0.4778, 0.2055]), torch.tensor([0.8040, 0.4523, 0.2539]))
      ])(back_im), \
      {
        "boxes": torch.as_tensor(boxes, dtype=torch.float),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
      }


transed, bl = DynamicSet()[1]
boxes = bl["boxes"]
labels = bl["labels"]
# show with bounding boxes
from torchvision.utils import draw_bounding_boxes
bb = draw_bounding_boxes(
  (transed*255).to(torch.uint8),
  #transed.to(torch.uint8),
  boxes,
  [L[l] for l in labels])
import matplotlib.pyplot as plt
assert len(labels) == len(boxes)
plt.imshow(bb.permute((1,2,0)))
plt.show()

