import os
from time import time

from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import visual

print(tf.config.list_physical_devices())

pretrained = False
try:
  model = keras.models.load_model('./tfdynanet.pt')
  print('loaded model')
  pretrained = True
except Exception as e:
  print("couldn't load model", str(e))
  model = Sequential([
    layers.Rescaling(1./255, input_shape=(25,25,3)),
    layers.Flatten(input_shape=(25,25,3)),
    layers.Dense(1000, activation='relu'),
    layers.Dense(500, activation='relu'),
    layers.Dense(1),
  ])

loss_fn = keras.losses.BinaryCrossentropy(
    from_logits=True, name='binary_crossentropy')

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

print(model.summary())

if not pretrained:
  ds = keras.utils.image_dataset_from_directory(
    "/Users/lair/fun/waai/dataset/",
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    image_size=(25,25),
    batch_size=4)

  print("len(ds)*4", len(ds)*4)

  epochs=5
  hist = model.fit(ds, validation_data=ds, epochs=epochs)

  model.save('./tfdynanet.pt')

  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']

  epochs_range = range(1, epochs+1)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()


imgs = []
with os.scandir('/Users/lair/fun/waai/source/') as it:
  for entry in it:
    if entry.is_file() and entry.name.lower().endswith('.png'):
      imgs.append(visual.load(entry.path))

for i in trange(len(imgs)):
  tiles = visual.tile(imgs[i])
  for j in trange(len(tiles)):
    sig = tf.nn.sigmoid(model(tf.expand_dims(tiles[j], 0), training=False))
    if tf.squeeze(sig) > .95:
      visual.write_img(tiles[j], f"./top/", name=f"im{i}_{int(time()*1000000)}.png")

