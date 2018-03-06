from os import listdir
from os.path import isfile, join

import random

# For displaying images
from IPython.display import Image
from IPython.display import display

mypath = 'simpson'
onlyfiles = [mypath + "/" + f for f in listdir(mypath) if f.endswith(".jpg")]

random.shuffle(onlyfiles)

for file in onlyfiles[:10]:
    img = Image(file)
    display(img)

# Build our data and labels!
import pprint

import math

import numpy as np
import PIL.Image as Image

from keras.utils import to_categorical

max_size = np.zeros(shape=(2,))

data = list()
labels = list()

for file in onlyfiles:

    # Get image as a numpy array
    im = Image.open(file)

    imarray = np.array(im)
    # print(imarray.shape)

    # So we can pad the images so they are all the same size
    if imarray.shape[0] > max_size[0]:
        max_size[0] = imarray.shape[0]

    if imarray.shape[1] > max_size[1]:
        max_size[1] = imarray.shape[1]

    # Get the labels
    tokens = file.rsplit('_', 1)
    label = tokens[0].split('/')[1]
    # print(label)

    data.append(imarray)
    labels.append(label)

print('max width: %d, max height: %d' % (max_size[0], max_size[1]))

# Pad the images to max width and max height
data = np.asarray(data)
new_data = list()
for img in data:
    img_new = np.zeros(shape=(int(max_size[0]), int(max_size[1]), 3))
    img_new[:img.shape[0], :img.shape[1], :] = img
    new_data.append(img_new)

new_data = np.asarray(new_data)
print('data shape: ' + str(new_data.shape))

# Build the label encoding

lbl = 0
label_map = dict()
for label in labels:
    if label not in label_map:
        label_map[label] = lbl
        lbl += 1

encoded_labels = list()
for label in labels:
    encoded_labels.append(label_map[label])

bin_labels = to_categorical(encoded_labels)

# Split data into Training, Validation, and Test
num_dat = len(encoded_labels)

train_data, train_labels = new_data[: math.floor(0.7 * num_dat)], bin_labels[: math.floor(0.7 * num_dat)]
valid_data, valid_labels = new_data[math.floor(0.7 * num_dat): math.floor(0.9 * num_dat)], bin_labels[math.floor(
    0.7 * num_dat): math.floor(0.9 * num_dat)]
test_data, test_labels = new_data[math.floor(0.9 * num_dat):], bin_labels[math.floor(0.9 * num_dat):]

# Imports..
from keras.models import Sequential  # Allows us to modularly add layers with ease.
from keras.layers import Conv2D, Activation, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras import optimizers

model = Sequential()

# Add layers to the model

model.add(
    Conv2D(
        filters=3,
        kernel_size=5,
        activation='linear',
        strides=1,
        input_shape=(int(max_size[0]), int(max_size[1]), 3)
    )
)

model.add(
    BatchNormalization()
)

model.add(
    Activation('tanh')
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation='linear'
    )
)

model.add(
    BatchNormalization()
)

model.add(
    Activation('relu')
)

model.add(
    Dense(
        lbl,
        activation='softmax'
    )
)

model.compile(
    loss=losses.categorical_crossentropy,
    optimizer=optimizers.Adadelta(),
    metrics=['accuracy']
)

print('Train data shape: ' + str(train_data.shape) + ', Train labels shape: ' + str(train_labels.shape))
print('Valid data shape: ' + str(valid_data.shape) + ', Valid labels shape: ' + str(valid_labels.shape))
print('Test data shape: ' + str(test_data.shape) + ', Test labels shape: ' + str(test_labels.shape))

# Fit the model.

history = model.fit(
    x=train_data,
    y=train_labels,
    epochs=30,
    verbose=1,
    validation_data=(valid_data, valid_labels)
)
