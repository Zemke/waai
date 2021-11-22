# WAAI

First off is detecting weapons in open battlefield.

Currently dataset is >5,000 800x640 of immediately after (about 60 milliseconds) dynamite drop frame captures.

## Approach 1

1. **MobileNetV3-Large** to classify dynamites trained on augmentations of `dynamite.png`. I hope that's sufficient training material.
  - Horizontal and vertical shifting -- `RandomAffine(0, (.34, .07))(img)` has proven good.
  - Different backgrounds. Ideally similar to captures which is mostly a worms behind the dyna and some rather dark background.
  - Yellow spark at the top can go right and left.
  - Set random screenshots from the map with no dyna which classify as "none."
  - Top left is the dynamite countdown<sup>1</sup>
2. Tile the 600x800 captures and classify each tile which already has the target size. Best match is the tile with the dyna -- hopefully.
3. Now you have the bounding box and an image for classification. This is the training data for the final **Faster R-CNN model with a MobileNetV3-Large FPN backbone**.

The hope is that the initial **MobileNetV3-Large** classifier doesn't have to be that good because for each 800x640 image we know there's one dynamite and the tile with the biggest confidence has to be that one. And then that's a perfect original in-game instance of dynamite use that can be fed into the final model along with the bounding box.

MobileNetV3 is chosen over for example ResNet because it's lightweight/fast and the assumptions is that Worms Armageddon being 2D comic-style with strong lines is a rather easy classification task.

<sup>1</sup> This is only available in replays, not in live games. For the final use case of live weapon usage tracking the neural net should also be trained without it.

