#!/usr/bin/env python

import os

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import trange, tqdm
import numpy as np


STEP = 10
EPOCHS = 18

device = torch.device(
  'cuda:0' if torch.cuda.is_available() else
  'mps' if torch.backends.mps.is_available() else
  'cpu')

class SingleNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(nn.functional.relu(self.conv1(x)))
    x = self.pool(nn.functional.relu(self.conv2(x)))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


  def device(self):
    self.to(device)
    print(f"moved net to {device}")
    return self

def _do(net, dl_train, dl_valid, loss_fn, optim, train=False):
  net.train(train)
  dl = dl_train if train else dl_valid

  losses, accs = [], []
  losses_val, accs_val = [], []
  running_loss, running_acc = \
    torch.tensor(0., device=device), \
    torch.tensor(0., device=device)
  progr = tqdm(dl,
               leave=train,
               colour='yellow' if train else 'green',
               postfix=dict(loss="na", acc="na"))
  for i, (b, l) in enumerate(progr):
    if train:
      optim.zero_grad()

    b, l = b.to(device), l.to(device)

    y = net(b)
    loss = loss_fn(y.view(-1), l)

    if train:
      loss.backward()
      optim.step()

    running_loss += loss.detach()
    running_acc += 1 - ((torch.sigmoid(y.detach().squeeze()) - l).abs().sum() / len(l))

    stepped = False
    step_mod = (i+1) % STEP
    if step_mod == 0:
      divisor = STEP
      stepped = True
    elif len(dl) == (i+1):
      divisor = step_mod
      stepped = True

    if stepped:
      losses.append((running_loss / divisor).cpu())
      accs.append((running_acc / divisor).cpu())
      running_loss, running_acc = \
        torch.tensor(0., device=device), \
        torch.tensor(0., device=device)
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
  optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  loss_fn = nn.BCEWithLogitsLoss()

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
  res = torch.tensor([], device=device)
  with tqdm(total=len(dl), leave=False) as bar:
    for i, b in enumerate(dl):
      net.eval()
      y = net(b.to(device))
      res = torch.cat([res, y.squeeze().detach()])
      bar.update()
  return res.cpu().numpy()

@torch.no_grad()
def pred(net, img):
  net.eval()
  return net(img.unsqueeze(0).to(device)).squeeze().item()

def save(net, path):
  return torch.save(net.state_dict(), path)

def pretrained(path):
  net = SingleNet()
  net.load_state_dict(torch.load(path))
  return net

