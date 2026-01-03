#!/usr/bin/env python3

import sys
import os
from subprocess import run

FPS = os.getenv("FPS", "2")
exe = os.getenv("WAPATH")
if exe is None:
  print("WAPATH not set")
  sys.exit(1)

def gettime(l):
  h, m, s = [float(n) for n in l[1:l.index("]")].split(':')]
  return h * 60 * 60 + m * 60 + s

L = open(sys.argv[1], encoding="ISO-8859-1", errors="ignore").read().splitlines()
sm = 0
idx = 0
while idx < len(L):
  if "starts turn" in L[idx]:
    fires = []
    while 'ends turn' not in L[idx] and 'loses turn' not in L[idx]:
      if 'fires' in L[idx]:
        weap = L[idx][L[idx].index('fires')+len('fires')+1:]
        print(weap)
        fires.append((gettime(L[idx]), weap.replace(" ", "").lower()))
      idx += 1
    fires.append((gettime(L[idx+1]), 'end'))
    sm += fires[-1][0] - fires[0][0]
    for i in range(0, len(fires)-1):
      start, end = fires[i][0], fires[i+1][0]
      prefix = f"{sys.argv[1].split('.')[0]}_{fires[i][1]}_".lower()
      cmd = [
        exe,
        "/getvideo",
        '.'.join(sys.argv[1].split(".")[:-1]) + ".WAgame",
        str(FPS),
        str(start),
        str(end),
        "1920",
        "1080",
        prefix
      ]
      if os.getenv("DRY") is None:
        run(cmd)
      else:
        print(cmd)
  idx += 1
print('total time:', sm)
print('frames:', int(sm * float(FPS) + 1))

