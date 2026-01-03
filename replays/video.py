#!/usr/bin/env python3

import sys
import os
import subprocess

FPS = os.getenv("FPS", "2")
WAPATH = os.getenv("WAPATH")
if WAPATH is None:
  print("WAPATH not set")
  sys.exit(1)
file = '.'.join(sys.argv[1].split(".")[:-1])
L = open(sys.argv[1], encoding="ISO-8859-1", errors="ignore").read().splitlines()

def gettime(l):
  h, m, s = [float(n) for n in l[1:l.index("]")].split(':')]
  return h * 60 * 60 + m * 60 + s

def setcmd(start, end, prefix):
  return [
    WAPATH + "/WA.exe",
    "/getvideo",
    file + ".WAgame",
    str(FPS),
    str(start),
    str(end),
    "1920",
    "1080",
    prefix
  ]

def fmtweap(weap):
  return weap.replace(" ", "").lower()

def run(cmd):
  if os.getenv("DRY") is None:
    subprocess.run(cmd)
  else:
    print(cmd)

sm = 0
idx = 0
while idx < len(L):
  if "starts turn" in L[idx]:
    fires = []
    while 'ends turn' not in L[idx] and 'loses turn' not in L[idx]:
      if 'fires' in L[idx]:
        weap = L[idx][L[idx].index('fires')+len('fires')+1:]
        fires.append((gettime(L[idx]), fmtweap(weap)))
      idx += 1
    fires.append((gettime(L[idx+1]), 'end'))
    sm += fires[-1][0] - fires[0][0]
    for i in range(0, len(fires)-1):
      start, end = fires[i][0], fires[i+1][0]
      prefix = f"{file}_{fires[i][1]}_".lower()
      run(setcmd(start, end, prefix))
  idx += 1
print('total time:', sm)
print('frames:', int(sm * float(FPS) + 1))

# TODO flamelets

# weapons that a worm can hold in her hand
#  almost looks like it's being used
weaps = [
  "Baseball Bat",
  "Battle Axe",
  "Blow Torch",
  "Pneumatic Drill",
]

for l in L:
  if 'fires' not in l:
    continue
  for weap in weaps:
    if weap in l:
      end = gettime(l)
      start = end - 5
      prefix = f"{file}_hand_{fmtweap(weap)}_"
      run(setcmd(start, end, prefix))

