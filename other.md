# What

Looking at a replay or live gameplay of Worms Armageddon, how to get to know which weapons were used?

# Goal

The end result is automatize the tool used in [this](https://www.twitch.tv/videos/814304436) video at the top to track used weapons.

# How

Besides all the rather manual approaches like bare-bones sound and image recognition the obvious idea is Machine Learning.

Reciting the manual approach: Some weapons make different sounds when used. This could be one way to the used weapon. \
Doesn't work for all and it probably gets complicated the deeper you dig.

# Approach

Machine Learning

Start finding the usage of a single weapon. For that I choose **shotgun**. \
It has a simple starting and ending sequence unlike ropes which can be shot in the air which wouldn't actually account for using it. Or worm select is used and then you're circling through your worms which. \
The weapon is also always visible in the worm's hands when used, the sound it makes is distinct in the game. Also it's used quite frequently which should make for more test data.

## Training

I have no idea so far. My naive Machine Learning approach goes something like this:

1. Choose an appropriate Neural Network design for the task.
2. Let it look at replays.
3. At the right time tell it, "That was as shotgun."
4. Profit.

### Training data

Given the tool [WAaaS](https://waaas.zemke.io) it gives at what time of the replay which weapon is used.

