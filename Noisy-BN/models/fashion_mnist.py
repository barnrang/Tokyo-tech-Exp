import os
import sys

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(HOME_PATH)

import tensorflow as tf
from tensorflow import keras

from layers.noisy_bn import NoisyBatchNormalization

def get_model(noisy=False, alpha=0.01, clear_session=True):
    use_batchnorm = keras.layers.BatchNormalization
    if noisy:
        use_batchnorm = lambda: NoisyBatchNormalization(alpha=alpha)

    if clear_session:
        tf.keras.backend.clear_session()

    input_node = keras.layers.Input((28,28))
    expand = keras.layers.Lambda(lambda x: tf.expand_dims(tf.cast(x, tf.float32) / 255., axis=-1))(input_node)

    conv1 = keras.layers.Conv2D(32, (3,3))(expand)
    batch1 = use_batchnorm()(conv1)
    pool1 = keras.layers.MaxPool2D((2,2))(batch1)

    conv2 = keras.layers.Conv2D(32, (3,3))(pool1)
    batch2 = use_batchnorm()(conv2)
    pool2 = keras.layers.MaxPool2D((2,2))(batch2)

    conv3 = keras.layers.Conv2D(32, (3,3))(pool2)
    batch3 = use_batchnorm()(conv3)
    pool3 = keras.layers.MaxPool2D((2,2))(batch3)

    av1 = keras.layers.GlobalAveragePooling2D()(pool3)

    h1 = keras.layers.Dense(512)(av1)
    batch4 = use_batchnorm()(h1)
    relu1 = keras.layers.Activation('relu')(batch4)

    output_h = tf.keras.layers.Dense(10)(relu1)
    output = tf.keras.layers.Activation('softmax')(output_h)

    return keras.Model(input_node, output)
