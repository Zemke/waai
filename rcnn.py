#!/usr/bin/env python

import os
import sys
from time import sleep

import torch
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import json
from tqdm import trange, tqdm

import multinet
import cropset
import dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")


def create_net():
  net = multinet.pretrained("./model_multinet.pt")
  backbone = net.features
  backbone.out_channels = 20
  num_classes = net.classifier[-1].out_features

  anchor_generator = AnchorGenerator(sizes=((10,30,40,),), aspect_ratios=((1.,.5,),))
  roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=15, sampling_ratio=1)

  return FasterRCNN(
    backbone, num_classes=num_classes,
    image_mean=dataset.MEAN, image_std=dataset.STD,
    min_size=640, max_size=1920,
    rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


def pretrained(path):
  model = create_net()
  model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
  print(f'loaded model {path} into CPU')
  return model


def train(model):
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=.005, momentum=0.9, weight_decay=0.0005)

  model.to(device)

  STEP = 2
  epochs = int(os.getenv('EPOCHS', 200))
  print(f'training {epochs} epochs')

  mlosses = []
  for epoch in trange(epochs, position=1):
    if (epoch+1) % 120 == 0:
      print("relax")
      sleep(10)  # GPU relaxation time

    r_loss = 0.
    for i, (img, l) in enumerate((progr := tqdm(dl, position=0))):
      img = list(i.to(device) for i in img)
      l = [{k: v.to(device) for k, v in t.items()} for t in l]
      model.train()
      loss_dict = model(img, l)
      losses = sum(loss for loss in loss_dict.values())
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      r_loss += losses
      log = False
      if (i+1) % STEP == 0:
        divisor = STEP
        log = True
      elif i+1 == len(dl):
        divisor = (i+1) % STEP
        log = True
      if log:
        mlosses.append(r_loss/divisor)
        progr.set_description(f"epoch:{epoch+1} loss:{mlosses[-1]:.4f}")
        r_loss = 0.

  return [i.detach().item() for i in mlosses]


def plot_loss(losses):
  plt.figure(figsize=(15,10))
  ax = plt.plot(losses)
  plt.ylim(top=.6)
  plt.show()


def infer(img, model):
  with torch.no_grad():
    model.eval()
    return model(F.to_tensor(img).unsqueeze(0))[0]


def plot_infer(y, img, thres):
  topk = y['scores'] > thres
  plt.figure(figsize=(16,15))

  labels = [dataset.CLASSES[i] for i in y['labels'][topk]]
  bb = draw_bounding_boxes(
    (F.to_tensor(img)*255).to(torch.uint8),
    boxes=y['boxes'][topk],
    labels=labels)
  plt.imshow(bb.permute(1,2,0))
  plt.title(list(zip(torch.round(y['scores'][topk] * 100).to(int).tolist(), labels)))
  plt.show()


def write_createml(targ_name, y, thres):
  targ_dir = os.getenv('TDIR', './labelling/target')
  dest = os.path.join(targ_dir, targ_name+'.json')
  if os.path.exists(dest):
    print(f'skipping json because {dest} exists')
    return

  j = {}
  j["image"] = targ_name+'.png'
  j["annotations"] = []
  topk = y['scores'] > thres
  for i, box in enumerate(y['boxes'][topk]):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    annot = {
      "coordinates": {
        "x": round((x1 + w / 2).item()),
        "y": round((y1 + h / 2).item()),
        "height": h.item(),
        "width": w.item()
      },
      "label": dataset.CLASSES[(y['labels'][i])]
    }
    j["annotations"].append(annot)

  with open(dest, 'w') as f:
    json.dump([j], f)
  return j


if __name__ == '__main__':
  if len(sys.argv) <= 1:
    model = create_net()

    batch_size = os.getenv('BATCH', 4)
    print(f"batch_size is {batch_size}")
    dl = cropset.load(cropset.CropSet().augment(), batch_size=batch_size)
    print('dataset length', len(dl.dataset))

    #model.load_state_dict(torch.load("./ubernet_100.pt"))
    losses = train(model)

    if os.getenv('PLOTLOSS') == '1':
      plot_loss(losses)

    print('save to ./ubernet.pt? [Y/n]', end=' ')
    if not input().strip().lower().startswith('n'):
      torch.save(model.state_dict(), "./ubernet.pt")
      print('saved')

  else:
    pretr_path = sys.argv[1]
    if not os.path.exists(pretr_path) or not pretr_path.endswith('.pt'):
      print(pretr_path, 'does not exist or is not a valid model')
      sys.exit(1)

    model = pretrained(pretr_path)
    paths = sys.argv[2:]
    for p in paths:
      print('f', p)
    for path in (progr := tqdm(paths)):
      progr.set_description(path[-20:])
      img = Image.open(path).convert('RGB')
      y = infer(img, model)
      thres = float(os.getenv('THRES', .8))
      if len(paths) == 1 or os.getenv('PLOTINFER') == '1':
        print(f"threshold score is {thres}")
        plot_infer(y, img, thres)
      if os.getenv('CREATEML') == '1':
        write_createml(path.split('/')[-1][:-4], y, thres)

