import argparse
import os
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.)
    parser.add_argument('--epochs', dest='epochs', type=int, default=1)
    #parser.add_argument('--suffix', dest='suffix', default='resnet')
    parser.add_argument('--gpu', dest='gpu', type=int, default=1)
    parser.add_argument('--train_size', dest='train_size', type=int, default=200)
    parser.add_argument('--reg_type', dest='reg', default='db')
    parser.add_argument('--model_type', dest='model_type', default='resnet')
    parser.add_argument('--data', dest='data', default='mnist')
    parser.add_argument('--seed', dest='seed', type=int, default=666)

    return parser.parse_args()

args = parser()
save_file_name = f'data/{args.data}_{args.reg}_{args.train_size}_{args.alpha}_{args.model_type}_{args.seed}.pk'
save_model_name = f'data/{args.data}_{args.reg}_{args.train_size}_{args.alpha}_{args.model_type}_{args.seed}.h5'
save_test_result_name = f'data/test_{args.data}_{args.reg}_{args.train_size}_{args.alpha}_{args.model_type}_{args.seed}.pk'

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import pickle
import numpy as np
np.random.seed(args.seed)
import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, Activation
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from resnet import resnet_v1
from mnist_helper import *
from small_model import mnist_small

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if args.data == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif args.data == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
else:
    raise f'No data named {args.data}'
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

if args.data == 'mnist':
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

idx = np.argsort(y_train)
x_train = x_train[idx]
y_train = y_train[idx]

idxs_per_class = [np.squeeze(np.argwhere(y_train == i)) for i in range(10)]

train_size_per_class = args.train_size
val_size_per_class = 5000

x_train_small = []
y_train_small = []

x_val = []
y_val = []


for i in range(10):
    perm_idx = np.random.permutation(len(idxs_per_class[i]))
    x_train_small.extend(x_train[idxs_per_class[i][perm_idx[:train_size_per_class]]])
    x_val.extend(x_train[idxs_per_class[i][perm_idx[train_size_per_class:train_size_per_class + val_size_per_class]]])
    y_train_small.extend(y_train[idxs_per_class[i][perm_idx[:train_size_per_class]]])
    y_val.extend(y_train[idxs_per_class[i][perm_idx[train_size_per_class:train_size_per_class + val_size_per_class]]])

x_train_small = np.array(x_train_small)
x_val = np.array(x_val)


train_datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True,
            horizontal_flip=True,)
val_datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True)

train_datagen.fit(x_train_small)
val_datagen.mean = train_datagen.mean
val_datagen.std = train_datagen.std
batch_size=32
val_gen = val_datagen.flow(x_test, to_categorical(y_test), shuffle=False, batch_size=batch_size)
# Initiate model
tf.keras.backend.clear_session()
model = load_model(save_model_name)
result = model.evaluate_generator(val_gen)
print(result)
with open(save_test_result_name, 'wb') as f:
    pickle.dump(result, f)
