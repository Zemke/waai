#!/usr/bin/env python3

import os
import sys
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
from torchvision.utils import draw_bounding_boxes

from collate import collate_fn

L = ['worm', 'mine', 'barrel', 'dynamite']

class DynamicSet(Dataset):
  def __init__(self):
    self.transform = T.Compose([
      T.PILToTensor(),
      v2.ToDtype(torch.float32, scale=True),
      # no norm in rcnn
      #T.Normalize(torch.tensor([0.8580, 0.4778, 0.2055]), torch.tensor([0.8040, 0.4523, 0.2539]))
    ])
    self.S = [30, 25, 40, 25]
    self.M = [Image.open('target/' + f).convert("RGBA") for f in listdir("target") if isfile(join("target", f)) and f.split(".")[-1] == "png"]
    self.W = [Image.open(c).convert("RGBA").resize((self.S[0], self.S[0])) for c in ['worms/' + f for f in listdir("worms") if isfile(join("worms", f)) and f.split(".")[-1] == "png"]]
    self.C = [Image.open(c).convert("RGBA").resize((self.S[t+1], self.S[t+1])) for t,c in enumerate([
      "weapons_alpha/mine.png",
      "weapons_alpha/barrel.png",
      "weapons_alpha/dynamite.png",
    ])]

  def __len__(self):
    return len(self.M)*100

  def __getitem__(self, idx):
    back_im = self.M[randrange(len(self.M))].copy()
    wi = 0
    SP = 45
    height, width, _ = np.array(back_im).shape
    a = Image.new("RGBA", back_im.size, (0,0,0,0))
    boxes, labels = [], []
    for y in range(SP-30, height-SP, SP):
      for x in range(SP-30, width-SP, SP):
        im2 = self.W[randrange(len(self.W))] if (t := randrange(4)) == 0 else self.C[t-1]
        wi += 1
        a.paste(im2, (x1 := x+randrange(10), y1 := y+randrange(10)))
        labels.append(t)
        boxes.append([x1, y1, x1+self.S[t], y1+self.S[t]])
    back_im = Image.alpha_composite(back_im, a).convert("RGB")
    return self.transform(back_im), {
      "boxes": torch.as_tensor(boxes, dtype=torch.float32),
      "labels": torch.as_tensor(labels, dtype=torch.int64),
    }


"""
transed, bl = DynamicSet()[1]

# show with bounding boxes
boxes = bl["boxes"]
labels = bl["labels"]
bb = draw_bounding_boxes(
  (transed*255).to(torch.uint8),
  boxes,
  [L[l] for l in labels])
F.to_pil_image(bb).show()
"""


from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn


def infer(img, model):
  with torch.no_grad():
    model.eval()
    to = T.Compose([
          T.PILToTensor(),
          v2.ToDtype(torch.float32, scale=True),
          # no norm in rcnn
          #T.Normalize(torch.tensor([0.8580, 0.4778, 0.2055]), torch.tensor([0.8040, 0.4523, 0.2539]))
        ])(img)

    return model(to.unsqueeze(0))[0]
    #return model(F.to_tensor(img).unsqueeze(0))[0]


def draw_bb(y, img, thres):
  topk = y['scores'] > thres
  labels = [L[i] for i in y['labels'][topk]]
  bb = draw_bounding_boxes(
    (F.to_tensor(img)*255).to(torch.uint8),
    boxes=y['boxes'][topk],
    labels=labels)
  return bb.permute(1,2,0).detach().numpy()


def output_img(y, img, thres, dest):
  img = Image.fromarray(draw_bb(y, img, thres))
  return img.save(dest)


if __name__ == '__main__':
  device = os.getenv('GPU', 'cpu')
  print(device)


  anchor_generator = AnchorGenerator(sizes=((10,30,40,),), aspect_ratios=((1.,.5,),))
  roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=15, sampling_ratio=1)

  model = fasterrcnn_mobilenet_v3_large_320_fpn(
    num_classes=len(L)+1,
    image_mean=(.5,.5,.5), image_std=(.5,.5,.5),  # TODO
    min_size=640, max_size=1920,
    #rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler)

  if True:  # TODO infer
    path = sys.argv[1]
    print('path', path)
    img = Image.open(path).convert('RGB')
    print(np.array(img).shape)
    model.load_state_dict(torch.load("./dynamicnet.pt", map_location=torch.device(device)))
    y = infer(img, model)
    thres = float(os.getenv('THRES', .8))
    output_img(y, img, thres, 'dest/res.png')
    assert False  # TODO infer

  #def collate_fn(batch):
  #  return tuple(zip(*batch))

  print('batch size ' + str(batch_size := int(os.getenv('BATCH', 4))))
  dl = DataLoader(
    DynamicSet(),
    #num_workers=4, persistent_workers=True,
    batch_size=batch_size, collate_fn=collate_fn)

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=.005, momentum=0.9, weight_decay=0.0005)

  model.to(device)
  model.train()

  epochs = int(os.getenv('EPOCHS', 200))
  print(f'training {epochs} epochs')

  for epoch in range(epochs):
    for i, (img, l) in enumerate(dl):
      img = torch.stack(img).to(device)
      l = [{k: v.to(device) for k, v in t.items()} for t in l]

      loss_dict = model(img, l)
      losses = sum(loss for loss in loss_dict.values())
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      print(losses.item())

      torch.save(model.state_dict(), "./dynamicnet.pt")

