#!/usr/bin/env python3

import os

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import trange, tqdm
import numpy as np


STEP = 5
EPOCHS = int(os.getenv("EPOCHS", 20))

device = torch.device(
  'cuda' if torch.cuda.is_available() else
  'mps' if torch.backends.mps.is_available() else
  'cpu') if os.getenv("GPU", False) == "1" else "cpu"

class MultiNet(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.num_classes = num_classes
    self.conv1 = nn.Conv2d(3, 10, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(10, 20, 3)
    self.conv_bn = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(20 * 5 * 5, 300)
    self.fc_bn = nn.BatchNorm1d(300)
    self.fc2 = nn.Linear(300, 200)
    self.fc3 = nn.Linear(200, 100)
    self.fc4 = nn.Linear(100, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv_bn(self.conv2(x))))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc_bn(self.fc1(x)))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
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
  losses_pc, accs_pc = [], []
  running_loss, running_acc = \
    torch.tensor(0., device=device), \
    torch.tensor(0., device=device)
  running_acc_pc, running_loss_pc = \
    torch.zeros(net.num_classes, device=device), \
    torch.zeros(net.num_classes, device=device)
  progr = tqdm(dl,
               leave=train,
               colour='yellow' if train else 'green',
               postfix=dict(loss="na", acc="na"))
  step = len(dl) // STEP + 1
  for i, (b, l) in enumerate(progr):
    if train:
      optim.zero_grad()

    b, l = b.to(device), l.to(device)

    y = net(b)
    loss = loss_fn(y, l)
    loss_mean = loss.mean()

    if train:
      loss_mean.backward()
      optim.step()

    running_loss += loss_mean.detach()
    running_acc += torch.count_nonzero(y.argmax(axis=1) == l) / len(b)

    if not train:
      # TODO optim for GPU
      for k in range(net.num_classes):
        clazz = torch.tensor(k, device=device)
        y_argmax = y.argmax(axis=1)
        for_clazz = (l == clazz).bitwise_or(y_argmax == clazz)
        running_acc_pc[k] += (y_argmax == l)[for_clazz].float().mean()
        running_loss_pc[k] += loss[for_clazz].mean()

    stepped = False
    step_mod = (i+1) % step
    if step_mod == 0:
      divisor = step
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
      if not train:
        losses_pc.append((running_loss_pc / divisor).cpu())
        accs_pc.append((running_acc_pc / divisor).cpu())
        running_acc_pc, running_loss_pc = \
          torch.zeros(net.num_classes, device=device), \
          torch.zeros(net.num_classes, device=device)
      if train and os.environ.get("TRAINONLY") != '1':
        with torch.no_grad():
          validres = _do(net, dl_train, dl_valid, loss_fn, None)
          losses_val.append(sum(validres[0])/len(validres[0]))
          accs_val.append(sum(validres[1])/len(validres[1]))
          losses_pc.append(sum(validres[2])/len(validres[2]))
          accs_pc.append(sum(validres[3])/len(validres[3]))
        progr.set_postfix(
            loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.2f}",
            val_loss=f"{losses_val[-1]:.4f}", val_acc=f"{accs_val[-1]:.2f}")
      else:
        progr.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.2f}")

  if train:
    return (losses, accs), (losses_val, accs_val), (losses_pc, accs_pc)
  else:
    return losses, accs, losses_pc, accs_pc


def train(net, dl_train, dl_valid=None):
  optim = torch.optim.Adam(net.parameters(), lr=1e-3)
  loss_fn = nn.CrossEntropyLoss(reduction='none')

  trainloss, trainacc = [], []
  validloss, validacc = [], []
  pcloss, pcacc = [], []
  for epoch in range(EPOCHS):
    print(f"{epoch+1:02} of {EPOCHS:02}")
    trainres, validres, pcres = \
        _do(net, dl_train, dl_valid or dl_train, loss_fn, optim, True)
    trainloss.extend(trainres[0])
    trainacc.extend(trainres[1])
    validloss.extend(validres[0])
    validacc.extend(validres[1])
    pcloss.extend(pcres[0])
    pcacc.extend(pcres[1])

  return (trainloss, trainacc), (validloss, validacc), (pcloss, pcacc)


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
  state_dict = torch.load(path, map_location=torch.device('cpu'))
  num_classes = len(state_dict[next(reversed(state_dict))])
  print(f'loaded module has {num_classes} classes')
  net = MultiNet(num_classes)
  net.load_state_dict(state_dict)
  return net

