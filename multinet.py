#!/usr/bin/env python3

import os

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import trange, tqdm
import numpy as np


STEP = 4
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
    self.fc1 = nn.Linear(20 * 5 * 5, 300)
    self.fc2 = nn.Linear(300, 200)
    self.fc3 = nn.Linear(200, 100)
    self.fc4 = nn.Linear(100, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

  def device(self):
    self.to(device)
    print(f"moved net to {device}")
    return self


def _do(net, dl_train, dl_test, loss_fn, optim, train):
  net.train(train)
  dl = dl_train if train else dl_test

  losses, accs = [], []
  losses_test, accs_test = [], []
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
      if train:
        with torch.no_grad():
          with dl_test.dataset.dataset.skip_augment():
            testres = _do(net, dl_train, dl_test, loss_fn, None, False)
          losses_test.append(sum(testres[0])/len(testres[0]))
          accs_test.append(sum(testres[1])/len(testres[1]))
          losses_pc.append(sum(testres[2])/len(testres[2]))
          accs_pc.append(sum(testres[3])/len(testres[3]))
        progr.set_postfix(
            loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.2f}",
            test_loss=f"{losses_test[-1]:.4f}", test_acc=f"{accs_test[-1]:.2f}")
      else:
        progr.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.2f}")

  if train:
    return (losses, accs), (losses_test, accs_test), (losses_pc, accs_pc)
  else:
    return losses, accs, losses_pc, accs_pc


def train(net, dl_train, dl_test):
  optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  loss_fn = nn.CrossEntropyLoss(reduction='none')

  mn_loss = torch.inf
  trainloss, trainacc = [], []
  testloss, testacc = [], []
  pcloss, pcacc = [], []
  for epoch in range(EPOCHS):
    print(f"{epoch+1:02} of {EPOCHS:02}")
    trainres, testres, pcres = _do(net, dl_train, dl_test, loss_fn, optim, True)

    trainloss.extend(trainres[0])
    trainacc.extend(trainres[1])
    testloss.extend(testres[0])
    testacc.extend(testres[1])
    pcloss.extend(pcres[0])
    pcacc.extend(pcres[1])

    if testloss[-1] < mn_loss:
      print('saving', name := f"model_{os.getenv('WEAPON', 'all')}.pt")
      torch.save(net.state_dict(), os.path.join('.', name))
      mn_loss = testloss[-1]

  return (trainloss, trainacc), (testloss, testacc), (pcloss, pcacc)


@torch.no_grad()
def pred_capture(net, tiles):
  net.eval()
  return F.softmax(net(tiles.to(device)), dim=1).numpy(force=True)


@torch.no_grad()
def pred(net, img):
  net.eval()
  return F.softmax(net(img.unsqueeze(0).to(device)).squeeze(), 0)


def pretrained(path):
  state_dict = torch.load(path, map_location=torch.device('cpu'))
  num_classes = len(state_dict[next(reversed(state_dict))])
  print(f'loaded module has {num_classes} classes')
  net = MultiNet(num_classes)
  net.load_state_dict(state_dict)
  return net

