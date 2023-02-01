# [WAAI](https://waai.zemke.io)

## Faster R-CNN Object Recognition

![Object Recognition Bounding Boxes](https://raw.githubusercontent.com/Zemke/waai/main/internet/img/recog.png)

[Blog](https://waai.zemke.io/object-recognition) • [Demo](https://www.youtube.com/watch?v=3sq1OArzWF8)

## TODO

- All images in the dataset originate from SingleNet.
  - Therefore `bg` from SingleNet runs should remain associated to the class.
    I.e. when SingleNet classifies sheep,
    `bg` from that inferencing should remain associated to the sheep.
    There could be a directory structure like `dataset/sheep/{0,1}/*.png`.
- Git LFS for dataset and models?
  - `alllogs`, `allreplays`
  - Keep all captures from `/getvideo` runs.
    They can be used to train the RPN later.
    Keep association to weapon.
  - Backups, too.
- Environment variables:
  - epochs
  - batch size
  - CPU/GPU


# Weapons

There are **26** weapons that reduce ammo
and of which there is enough data from replays.

![Weapon](weapons/138.bmp)
![Weapon](weapons/140.bmp)
![Weapon](weapons/141.bmp)
![Weapon](weapons/144.bmp)
![Weapon](weapons/146.bmp)
![Weapon](weapons/147.bmp)
![Weapon](weapons/150.bmp)
![Weapon](weapons/153.bmp)
![Weapon](weapons/155.bmp)
![Weapon](weapons/157.bmp)
![Weapon](weapons/158.bmp)
![Weapon](weapons/159.bmp)
![Weapon](weapons/162.bmp)
![Weapon](weapons/167.bmp)
![Weapon](weapons/169.bmp)
![Weapon](weapons/170.bmp)
![Weapon](weapons/171.bmp)
![Weapon](weapons/174.bmp)
![Weapon](weapons/177.bmp)
![Weapon](weapons/184.bmp)
![Weapon](weapons/187.bmp)
![Weapon](weapons/188.bmp)
![Weapon](weapons/189.bmp)
![Weapon](weapons/299.bmp)
![Weapon](weapons/300.bmp)
![Weapon](weapons/353.bmp)


