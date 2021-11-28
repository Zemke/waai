#!/usr/bin/env python

import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from tqdm import trange
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
  for epoch in range(12):
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

