#!/usr/bin/env python3

import os

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import Counter

device = torch.device(
  'cuda' if torch.cuda.is_available() else
  'mps' if torch.backends.mps.is_available() else
  'cpu') if os.getenv("GPU", False) == "1" else "cpu"
#print("device is", device)


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

    self.to(device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x


class Trainer:
  lr = float(os.getenv("LR", 1e-3))
  epochs = int(os.getenv("EPOCHS", 20))

  def __init__(self, net, classes, dataloader, tester):
    print("lr:", self.lr)
    print("epochs:", self.epochs)

    self.net = net
    self.classes = classes
    self.dataloader = dataloader
    self.tester = tester

    self.net = MultiNet(len(self.classes)).to(device)
    self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
    self.loss_fn = nn.CrossEntropyLoss()

  def __call__(self):
    step = max(1, len(self.dataloader) // 10)

    self.metric = {k: [] for k in ["loss", "acc", "test_loss", "test_acc", "test_acc_pc"]}
    mn_test_loss = 9e+9
    best_conf_mat = None
    running_acc = torch.tensor(0., device=device)
    running_loss = torch.tensor(0., device=device)
    stepped = torch.tensor(0., device=device)

    for epoch in range(1, self.epochs + 1):
      self.net.train()

      for i, (x, l) in (progr := tqdm(
        enumerate(self.dataloader),
        total=len(self.dataloader),
        colour='yellow',
        postfix=(pf := {}),
        desc=f"{epoch:02}"
      )):
        self.optim.zero_grad()

        x, l = x.to(device), l.to(device)

        y = self.net(x)
        loss = self.loss_fn(y, l)

        loss.backward()
        self.optim.step()

        running_acc += (l == y.argmax(1)).sum() / l.numel()
        running_loss += loss
        stepped += 1

        if (last_batch := i+1 == len(self.dataloader)) or (i+1) % step == 0:
          self.metric["loss"].append((running_loss / stepped).item())
          self.metric["acc"].append((running_acc / stepped).item())

          running_acc -= running_acc
          running_loss -= running_loss
          stepped -= stepped

          if last_batch:
            test_metric, conf_mat = self.tester(self.net)
            for k,v in test_metric.items():
              self.metric[k].append(test_metric[k])

            pf["test_loss"] = f"{self.metric['test_loss'][-1]:.4f}"
            pf["test_acc"] = f"{self.metric['test_acc'][-1]:.2f}"
            progr.set_postfix(pf)

            if self.metric["test_loss"][-1] < mn_test_loss:
              progr.colour = 'green'
              mn_test_loss = self.metric["test_loss"][-1]
              best_conf_mat = conf_mat
              torch.save(self.net.state_dict(), "./model.pt")
            else:
              progr.colour = 'red'

          pf["loss"] = f"{self.metric['loss'][-1]:.4f}"
          pf["acc"] = f"{self.metric['acc'][-1]:.2f}"
          progr.set_postfix(pf)

    return self.metric, best_conf_mat


class Tester:
  def __init__(self, dataset, counts):
    self.dataset = dataset
    self.counts = counts
    self.weight = (reciprocal := (1 / torch.tensor(counts, device=device))) / sum(reciprocal)
    self.loss_fn = nn.CrossEntropyLoss(self.weight)

  @torch.no_grad()
  def __call__(self, net):
    with self.dataset.dataset.skip_augment():
      x, l = zip(*[(x, l) for x,l,_ in self.dataset])
    x = torch.stack(x).to(device)
    l = torch.tensor(l).to(device)
    y = net(x)
    pred = y.argmax(1).int()
    # confusion matrix
    M = torch.zeros((net.num_classes, net.num_classes), dtype=int)
    for pred_l in zip(pred, l):
      M[pred_l] += 1
    acc_pc = torch.tensor([M[c,c] / self.counts[c] for c in range(net.num_classes)])
    return \
      dict(
        test_loss=self.loss_fn(y, l).item(),
        test_acc=acc_pc.mean().item(),
        test_acc_pc=acc_pc.tolist(),
      ), M


@torch.no_grad()
def pred_capture(net, tiles):
  net.eval()
  return F.softmax(net(tiles.to(device)), dim=1).numpy(force=True)


@torch.no_grad()
def pred(net, img):
  net.eval()
  return F.softmax(net(img.unsqueeze(0).to(device)).squeeze(), 0).numpy(force=True)


def pretrained(path):
  state_dict = torch.load(path, map_location=torch.device('cpu'))
  num_classes = len(state_dict[next(reversed(state_dict))])
  print(f'loaded module has {num_classes} classes')
  net = MultiNet(num_classes)
  net.load_state_dict(state_dict)
  return net

