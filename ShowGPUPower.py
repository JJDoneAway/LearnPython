#this is a long running model creation.
# start it and check if the GPU is used via your system monitoring 

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers


import numpy as np


# configure GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)(nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#keras has already the full MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


network = models.Sequential()
network.add(layers.Dense(484, activation='sigmoid', input_shape=(28 * 28,)))
network.add(layers.Dense(289, activation='sigmoid'))
network.add(layers.Dense(10, activation='sigmoid'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#resape to 2D and float
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

history  = network.fit(train_images, train_labels, epochs=500, batch_size=1)

