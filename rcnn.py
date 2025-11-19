#!/usr/bin/env python

import os
import sys

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

enable_progr = os.getenv("PROGR", True) != "0"

use_dynamicset = False
if os.getenv("DYNAMICSET", False) == "1":
  import dynamicset as dataset
  use_dynamicset = True
else:
  import dataset as dataset
print('use_dynamicset', use_dynamicset)


device = torch.device(
  'cuda' if torch.cuda.is_available() else
  'mps' if torch.backends.mps.is_available() else
  'cpu') if os.getenv("GPU", False) == "1" else "cpu"


def create_net():
  if (bb_path := os.getenv("BACKBONE")):
    net = multinet.pretrained(bb_path)
  else:
    net = multinet.MultiNet(len(dataset.CLASSES))
  backbone = net.features
  backbone.out_channels = 20
  num_classes = net.classifier[-1].out_features
  print('num_classes', num_classes)

  anchor_generator = AnchorGenerator(sizes=((10,30,40,),), aspect_ratios=((1.,.5,),))
  roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=15, sampling_ratio=1)

  return FasterRCNN(
    backbone, num_classes=num_classes,
    image_mean=dataset.MEAN, image_std=dataset.STD,
    min_size=640, max_size=1920,
    rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
    box_detections_per_img=200)


def pretrained(path):
  model = create_net()
  model.load_state_dict(torch.load(path, map_location=torch.device(device)))
  print(f'loaded model {path} into {device}')
  return model


def train(model):
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=.008, momentum=0.9, weight_decay=0.0005)

  model.to(device)
  model.train()

  STEP = 2
  epochs = int(os.getenv('EPOCHS', 200))
  print(f'training {epochs} epochs')

  lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=epochs//4,
    gamma=.7,
  )

  mlosses = []
  mn_loss = None
  for epoch in trange(epochs, position=1, disable=not enable_progr):
    r_loss = 0.
    for i, (img, l) in enumerate((progr := tqdm(dl, position=0, disable=not enable_progr))):
      img = list(i.to(device) for i in img)
      l = [{k: v.to(device) for k, v in t.items()} for t in l]

      loss_dict = model(img, l)
      print(epoch, loss_dict);
      sys.stdout.flush()
      losses = sum(loss for loss in loss_dict.values())
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      r_loss += losses.item()
      log = False
      if (i+1) % STEP == 0:
        divisor = STEP
        log = True
      elif i+1 == len(dl):
        divisor = (i+1) % STEP
        log = True
      if log:
        mlosses.append(r_loss/divisor)
        progr.set_description(f"epoch:{epoch+1} loss:{mlosses[-1]:.4f} lr:{lr_scheduler.get_last_lr()}")
        r_loss = 0.
    lr_scheduler.step()
    if mn_loss is None or mlosses[-1] < mn_loss:
      torch.save(model.state_dict(), "./ubernet.pt")
      mn_loss = mlosses[-1]
      print(f"ubernet.pt saved at epoch {epoch+1} and loss {mn_loss}")

  return mlosses


def plot_loss(losses):
  plt.figure(figsize=(15,10))
  ax = plt.plot(losses)
  plt.ylim(top=.6)
  plt.savefig("lossplot.png")


def infer(img, model):
  with torch.no_grad():
    model.eval()
    return model(F.to_tensor(img).unsqueeze(0))[0]


def draw_bb(y, img, thres):
  topk = y['scores'] > thres
  labels = [dataset.CLASSES[i] for i in y['labels'][topk]]
  bb = draw_bounding_boxes(
    (F.to_tensor(img)*255).to(torch.uint8),
    boxes=y['boxes'][topk],
    labels=labels)
  return bb.permute(1,2,0).detach().numpy()

def plot_infer(y, img, thres):
  topk = y['scores'] > thres
  plt.figure(figsize=(16,15))
  labels = [dataset.CLASSES[i] for i in y['labels'][topk]]
  plt.imshow(draw_bb(y, img, thres))
  plt.title(list(zip(torch.round(y['scores'][topk] * 100).to(int).tolist(), labels)))
  plt.show()


def output_img(y, img, thres, dest):
  img = Image.fromarray(draw_bb(y, img, thres))
  return img.save(dest)


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
  if len(sys.argv) <= 1:  # train
    env_plotloss = os.getenv('PLOTLOSS') == '1'
    print("env_plotloss", env_plotloss)
    model = create_net()

    batch_size = int(os.getenv('BATCH', 4))
    print(f"batch_size is {batch_size}")
    if use_dynamicset:
      dl = dataset.load(
        dataset.DynamicSet(batch_size*100),
        batch_size=batch_size,
        pin_memory=str(device) == "cuda")
    else:
      dl = cropset.load(cropset.CropSet().augment(), batch_size=batch_size)
    print('dataset length', len(dl.dataset))

    if (pretr := os.getenv('PRETR')) != None:
      print("loading pretrained model:", pretr)
      model.load_state_dict(torch.load(pretr, map_location=device))
    losses = train(model)

    if env_plotloss:
      plot_loss(losses)
  else:
    pretr_path = sys.argv[1]
    if not os.path.exists(pretr_path) or not pretr_path.endswith('.pt'):
      print(pretr_path, 'does not exist or is not a valid model')
      sys.exit(1)
    model = pretrained(pretr_path)
    thres = float(os.getenv('THRES', .8))
    print(f"threshold score is {thres}")

    if len(sys.argv) == 3:  # infer single
      img = Image.open(sys.argv[2]).convert('RGB')
      y = infer(img, model)
      plot_infer(y, img, thres)
    else:  # mass infer
      paths = sys.argv[2:]
      output_dir = None
      if os.path.isdir(paths[0]):
        output_dir = paths[0]
        paths = paths[1:]
        print("outputting to path", output_dir)
      for path in (progr := tqdm(paths)):
        if output_dir is not None:
          assert os.path.exists(output_dir)
          dest = os.path.join(output_dir, os.path.basename(path))
          if not os.path.exists(dest):
            progr.set_description(path[-20:])
            img = Image.open(path).convert('RGB')
            y = infer(img, model)
            output_img(y, img, thres, dest)
        elif len(paths) == 1 or os.getenv('PLOTINFER') == '1':
          for p in paths:
            print('f', p)
          img = Image.open(path).convert('RGB')
          y = infer(img, model)
          plot_infer(y, img, thres)
          if os.getenv('CREATEML') == '1':
            write_createml(path.split('/')[-1][:-4], y, thres)

