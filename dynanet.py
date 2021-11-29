#!/usr/bin/env python

import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from tqdm import trange, tqdm
import numpy as np


STEP = 11


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


def _do(net, dataloader, train=False):
  optim = torch.optim.Adam(net.parameters(), lr=1e-5)
  loss_fn = nn.BCEWithLogitsLoss()

  net.train(train)

  losses, accs = [], []
  epochs = 5
  for epoch in range(epochs):
    running_loss, running_acc = 0., 0.
    progr = tqdm(dataloader,
                 desc=f"{epoch+1:02}/{epochs:02}",
                 colour='red' if train else 'green',
                 postfix=dict(loss="na", acc="na",
                              epoch=f"{epoch+1}/{epochs}"))
    for i, (b, l) in enumerate(progr):
      if train:
        optim.zero_grad()

      y = net(b)
      loss = loss_fn(y.view(-1), l)

      if train:
        loss.backward()
        optim.step()

      # accuracy
      y_sigmoid = nn.Sigmoid()(y.squeeze())
      stck = torch.stack((l, y_sigmoid.view(-1)), 1).detach().numpy()
      stck = np.diff(stck).reshape(-1) - 1
      stck[stck < -1] += 2
      acc = np.abs(stck).sum() / len(b)

      running_loss += loss.item()
      running_acc += acc
      progr.set_postfix(loss=f"{running_loss/(i+1):.4f}",
                        acc=f"{running_acc/(i+1):.2f}")

    losses.append(running_loss/len(dataloader))
    accs.append(running_acc/len(dataloader))

  return losses, accs


def train(net, dataloader):
  return _do(net, dataloader, True)


def valid(net, dataloader):
  with torch.no_grad():
    return _do(net, dataloader)


def eval(net, imgs):
  res = []
  with torch.no_grad():
    for i in trange(len(imgs), leave=False):
      net.eval()

      # standardize
      norm_img = Compose([ToTensor(), Resize((25,25)),])(imgs[i])
      try:
        std, mean = torch.std_mean(norm_img.unsqueeze(0), (0,2,3))
        std = torch.maximum(std, torch.full((3,), .0001))
        norm_img = Normalize(std=std, mean=std)(norm_img)
      except ValueError as e:  # std is 0
        p5 = torch.full((3,), .5)
        norm_img = Normalize(std=p5, mean=p5)(norm_img)

      r = net(norm_img.unsqueeze(0))
      res.append(r.squeeze().item())
  return res


def save(net, path):
  return torch.save(net.state_dict(), path)


def pretrained(path):
  net = DynaNet()
  net.load_state_dict(torch.load(path))
  return net

