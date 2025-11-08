import re
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w

plt.ion()
plt.style.use('dark_background')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

y = []
G = L = None
while True:
  with open('train.log', 'r') as f:
    for line in f.read().splitlines():
      if re.match("^[0-9]+ ", line) is None:
        continue
      if line not in y:
        print('plt', line)
        y.append(line)
  if len(y) < 1:
    continue
  z = [[float(n) for n in re.sub("[^0-9.]+", ",", re.sub('^[0-9]+ ', '', y1)).replace(",0,", ",")[1:-1].split(",")] for y1 in y]
  zt = np.transpose(z)
  labels = re.findall("'(.*?)'", y[0])
  if G is None:
    G = [None] * len(labels)
  colors = ['magenta', 'cyan', 'lime', 'yellow', 'red', 'orange', 'blue', 'green'][:len(labels)]
  for i,l in enumerate(labels):
    if G[i] is not None:
      G[i][0].remove()
    #G[i] = plt.plot(moving_average(zt[i], 15), label=l, color=colors[i])
    G[i] = plt.plot(zt[i], label=l, color=colors[i])
  if L is not None:
    L.remove()
  L = plt.legend([g for g, in G], labels)
  plt.pause(2)

