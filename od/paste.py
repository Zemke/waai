#!/usr/bin/env python3

import os
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

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from collate import collate_fn


S = [30, 25, 40, 25]
L = ['worm', 'mine', 'barrel', 'dynamite']
M = [Image.open('target/' + f).convert("RGBA") for f in listdir("target") if isfile(join("target", f)) and f.split(".")[-1] == "png"]
W = [Image.open(c).convert("RGBA").resize((S[0], S[0])) for c in ['worms/' + f for f in listdir("worms") if isfile(join("worms", f)) and f.split(".")[-1] == "png"]]
C = [Image.open(c).convert("RGBA").resize((S[t+1], S[t+1])) for t,c in enumerate([
  "weapons_alpha/mine.png",
  "weapons_alpha/barrel.png",
  "weapons_alpha/dynamite.png",
])]

class DynamicSet(Dataset):
  def __init__(self):
    self.transform = T.Compose([
      T.PILToTensor(),
      v2.ToDtype(torch.float32, scale=True),
      # no norm in rcnn
      #T.Normalize(torch.tensor([0.8580, 0.4778, 0.2055]), torch.tensor([0.8040, 0.4523, 0.2539]))
    ])

  def __len__(self):
    return len(M)*100

  def __getitem__(self, idx):
    im1 = M[randrange(len(M))]
    wi = 0
    SP = 45
    height, width, _ = np.array(im1).shape
    im1 = im1.convert("RGBA")
    back_im = im1.copy().convert("RGBA")
    boxes, labels = [], []
    for y in range(SP-30, height-SP, SP):
      for x in range(SP-30, width-SP, SP):
        a = Image.new("RGBA", back_im.size, (0,0,0,0))
        c = W[randrange(len(W))] if (t := randrange(4)) == 0 else C[t-1]
        im2 = c
        wi += 1
        a.paste(im2, (x1 := x+randrange(10), y1 := y+randrange(10)))
        back_im = Image.alpha_composite(back_im, a)
        labels.append(t)
        boxes.append([x1, y1, x1+S[t], y1+S[t]])
    back_im = back_im.convert("RGB")
    #back_im.show()
    return self.transform(back_im), {
      "boxes": torch.as_tensor(boxes, dtype=torch.float),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


"""
transed, bl = DynamicSet()[1]

# show with bounding boxes
boxes = bl["boxes"]
labels = bl["labels"]
from torchvision.utils import draw_bounding_boxes
bb = draw_bounding_boxes(
  (transed*255).to(torch.uint8),
  boxes,
  [L[l] for l in labels])
F.to_pil_image(bb).show()
"""


from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

if __name__ == '__main__':
  #model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
  model = fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=len(L)+1)

  anchor_generator = AnchorGenerator(sizes=((10,30,40,),), aspect_ratios=((1.,.5,),))
  roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=15, sampling_ratio=1)

  model = fasterrcnn_mobilenet_v3_large_320_fpn(
    num_classes=len(L)+1,
    image_mean=(.5,.5,.5), image_std=(.5,.5,.5),  # TODO
    min_size=640, max_size=1920,
    #rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler)

  #def collate_fn(batch):
  #  return tuple(zip(*batch))

  dl = DataLoader(
    DynamicSet(),
    num_workers=4, persistent_workers=True,
    batch_size=8, collate_fn=collate_fn)

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=.005, momentum=0.9, weight_decay=0.0005)

  device = 'cpu'
  model.to(device)
  model.train()

  epochs = int(os.getenv('EPOCHS', 200))
  print(f'training {epochs} epochs')

  for epoch in range(epochs):
    for i, (img, l) in enumerate(dl):
      img = list(i.to(device) for i in img)
      l = [{k: v.to(device) for k, v in t.items()} for t in l]

      loss_dict = model(img, l)
      losses = sum(loss for loss in loss_dict.values())
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      print(losses.item())

