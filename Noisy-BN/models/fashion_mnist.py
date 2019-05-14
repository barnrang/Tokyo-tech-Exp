import os
import sys

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(HOME_PATH)

import tensorflow as tf
from tensorflow import keras

from layers.noisy_bn import BetterNoisyBatchNormalization

def get_model(alpha=0., p=0.25, clear_session=True):
    feature_layers = [
        keras.layers.Conv2D(32,(2,2),padding='valid',activation='relu',input_shape=(28,28,1)),
        BetterNoisyBatchNormalization(alpha,p),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128,(2,2),padding='valid',activation='relu'),
        BetterNoisyBatchNormalization(alpha,p),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten()
    ]

    classify_layer = [
        keras.layers.Dense(128,activation='relu'),
        BetterNoisyBatchNormalization(alpha,p),
        keras.layers.Dense(64,activation='relu'),
        BetterNoisyBatchNormalization(alpha,p),
        keras.layers.Dense(10,activation='softmax')
    ]

    return keras.models.Sequential(feature_layers + classify_layer)

