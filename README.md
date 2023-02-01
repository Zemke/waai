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


## Weapons

There are **26** weapons that reduce ammo
and of which there is enough data from replays.

### Dropped

![Weapon](weapons/147.bmp)
![Weapon](weapons/162.bmp)

### Attached

Weapons that are visually attached to the worms using it.

![Weapon](weapons/138.bmp)
![Weapon](weapons/140.bmp)
![Weapon](weapons/141.bmp)
![Weapon](weapons/146.bmp)
![Weapon](weapons/157.bmp)
![Weapon](weapons/189.bmp)
![Weapon](weapons/174.bmp)
![Weapon](weapons/169.bmp)
![Weapon](weapons/299.bmp)

### Shot

![Weapon](weapons/155.bmp)
![Weapon](weapons/158.bmp)
![Weapon](weapons/171.bmp)

### Thrown

![Weapon](weapons/170.bmp)
![Weapon](weapons/144.bmp)
![Weapon](weapons/153.bmp)

### Released

Released and than having their own mind (animals).

![Weapon](weapons/159.bmp)
![Weapon](weapons/177.bmp)
![Weapon](weapons/188.bmp)

### Airborne

Air Strikes.

![Weapon](weapons/187.bmp)
![Weapon](weapons/167.bmp)

### Placed

Weapons places with a click pointer cursor.

![Weapon](weapons/150.bmp)
![Weapon](weapons/184.bmp)

### Other

![Weapon](weapons/300.bmp)
![Weapon](weapons/353.bmp)


