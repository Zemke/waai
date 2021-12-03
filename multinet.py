#!/usr/bin/env python

import os

import torch
from torch import nn
from torchvision.transforms import functional as F

from tqdm import trange, tqdm
import numpy as np

STEP = 15
EPOCHS = 3


# TODO network too big
class MultiNet(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.convs = nn.Sequential(
      nn.Conv2d(3, 16, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(16, 85, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6,6))
    self.classifier = nn.Sequential(
      nn.Dropout(p=.5),
      nn.Flatten(),
      nn.Linear(3060, 1000),
      nn.ReLU(inplace=True),
      nn.Dropout(p=.5),
      nn.Linear(1000, 500),
      nn.ReLU(inplace=True),
      nn.Dropout(p=.5),
      nn.Linear(500, num_classes)
    )

  def forward(self, x):
    x = self.convs(x)
    x = self.avgpool(x)
    #print(x.flatten(1).shape)
    x = self.classifier(x)
    return x


def _do(net, dl_train, dl_valid, loss_fn, optim, train=False):
  net.train(train)
  dl = dl_train if train else dl_valid

  losses, accs = [], []
  losses_val, accs_val = [], []
  running_loss, running_acc = 0., 0.
  progr = tqdm(dl,
               leave=train,
               colour='yellow' if train else 'green',
               postfix=dict(loss="na", acc="na"))
  for i, (b, l) in enumerate(progr):
    if train:
      optim.zero_grad()

    y = net(b)
    loss = loss_fn(y, l)

    if train:
      loss.backward()
      optim.step()

    # accuracy
    running_loss += loss.item()
    running_acc += torch.count_nonzero(y.argmax(axis=1) == l) / 4

    stepped = False
    step_mod = (i+1) % STEP
    if step_mod == 0:
      divisor = STEP
      stepped = True
    elif len(dl) == (i+1):
      divisor = step_mod
      stepped = True

    if stepped:
      losses.append(running_loss/divisor)
      accs.append(running_acc/divisor)
      running_loss, running_acc = 0., 0.
      if train and os.environ.get("TRAINONLY") != '1':
        with torch.no_grad():
          validres = _do(net, dl_train, dl_valid, loss_fn, None)
          losses_val.append(sum(validres[0])/len(validres[0]))
          accs_val.append(sum(validres[1])/len(validres[1]))
        progr.set_postfix(
            loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.2f}",
            val_loss=f"{losses_val[-1]:.4f}", val_acc=f"{accs_val[-1]:.2f}")
      else:
        progr.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.2f}")

  if train:
    return (losses, accs), (losses_val, accs_val)
  else:
    return losses, accs


def train(net, dl_train, dl_valid=None):
  optim = torch.optim.Adam(net.parameters(), lr=1e-3)
  loss_fn = nn.CrossEntropyLoss()

  trainloss, trainacc = [], []
  validloss, validacc = [], []
  for epoch in range(EPOCHS):
    print(f"{epoch+1:02} of {EPOCHS:02}")
    trainres, validres = \
        _do(net, dl_train, dl_valid or dl_train, loss_fn, optim, True)
    trainloss.extend(trainres[0])
    trainacc.extend(trainres[1])
    validloss.extend(validres[0])
    validacc.extend(validres[1])

  return (trainloss, trainacc), (validloss, validacc)


@torch.no_grad()
def pred_capture(net, dl):
  res = []
  for i, b in enumerate(dl):
    net.eval()
    y = net(b)
    res.extend(y.detach().numpy())
    break
  return res


@torch.no_grad()
def pred(net, img):
  net.eval()
  # TODO centralize transforms
  y = net(F.resize(F.to_tensor(img), (30,30)).unsqueeze(0))
  return y.squeeze().detach().numpy()


def save(net, path):
  return torch.save(net.state_dict(), path)


def pretrained(path):
  state_dict = torch.load(path)
  num_classes = len(state_dict[next(reversed(state_dict))])
  print(f'loaded module has {num_classes} classes')
  net = MultiNet(num_classes)
  net.load_state_dict(state_dict)
  return net

