#!/usr/bin/env python

import torch
from torch import nn
import numpy as np


class DynaNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.fc = nn.Sequential(
      nn.Linear(1875, 1000),
      nn.ReLU(),
      nn.Linear(1000, 500),
      nn.ReLU(),
      nn.Linear(500, 1),
    )

  def forward(self, x):
    x = self.flatten(x)
    x = self.fc(x)
    return x


def train(net, dataloader):
  optim = torch.optim.Adam(net.parameters(), lr=1e-5)
  loss_fn = nn.BCEWithLogitsLoss()

  step = 11
  losses, acc = [], []
  for epoch in range(8):  # TODO 12
    running_loss, running_acc = 0., 0.
    for i, (b, l) in enumerate(dataloader):
      net.train()
      optim.zero_grad()
      y = net(b)
      loss = loss_fn(y.squeeze(), l)
      loss.backward()
      optim.step()

      # loss and accuracy
      running_loss += loss.item()
      y_sigmoid = nn.Sigmoid()(torch.squeeze(y))
      stck = torch.stack((l, y_sigmoid), 1).detach().numpy()
      stck = np.diff(stck).squeeze() - 1
      stck[stck < -1] += 2
      running_acc += np.abs(stck).sum() / 4

      if i % step == step-1:
        losses.append(running_loss/step)
        acc.append(running_acc/step)
        print(f"{epoch:02}/{i:03} loss:{losses[-1]:.4f} acc:{acc[-1]:.2f}")
        running_loss, running_acc = 0., 0.

  return losses, acc


def eval(net, imgs, transforms):
  total = len(imgs)
  rr = []
  with torch.no_grad():
    for i in range(len(imgs)):
      net.eval()
      r = net(transforms(imgs[i]).unsqueeze(0))
      rr.append(r)
      if i % 100 == 0:
        print(f'{(i+1)/total*100:.0f}%', end='\r')
  print("100%", end='\r')
  print()
  return rr


def save(net, path):
  return torch.save(net.state_dict(), path)


def pretrained(path):
  net = DynaNet()
  net.load_state_dict(torch.load(path))
  return net

