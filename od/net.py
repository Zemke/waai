#!/usr/bin/env python3

classes = [
  'dynamite',
  'mine',
  'barrel',
  'cow',
  'sheep',
  'skunk',
  'jetpack',
  'chute',
  'worm',
]

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

#model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
model = fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=len(classes)+1)

