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
    parser.add_argument('--model_type', dest='model_type', default='small')
    parser.add_argument('--seed', dest='seed', type=int, default=666)
    parser.add_argument('--add_noise', dest='add_noise', default=False, action='store_true')
    parser.add_argument('--alpha_noise', dest='alpha_noise', type=float, default=0.)

    return parser.parse_args()

args = parser()

if args.add_noise and (args.reg != 'NI'):
    save_file_name = f'data/cifar10_{args.reg}_with_noise_{args.alpha_noise}_{args.train_size}_{args.alpha}_{args.model_type}_{args.seed}.pk'
    save_model_name = f'data/cifar10_{args.reg}_with_noise_{args.alpha_noise}_{args.train_size}_{args.alpha}_{args.model_type}_{args.seed}.h5'
else:
    save_file_name = f'data/cifar10_{args.reg}_{args.train_size}_{args.alpha}_{args.model_type}_{args.seed}.pk'
    save_model_name = f'data/cifar10_{args.reg}_{args.train_size}_{args.alpha}_{args.model_type}_{args.seed}.h5'

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import pickle
import numpy as np
np.random.seed(args.seed)
from tensorflow.random import set_seed
set_seed(args.seed)
import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, Activation
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from resnet import resnet_v1
from cifar_helper import *
from small_model import cifar_small

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

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

batch_size = 32

train_gen = train_datagen.flow(x_train_small, to_categorical(y_train_small), batch_size=batch_size)
val_gen = val_datagen.flow(x_val, to_categorical(y_val), shuffle=False, batch_size=batch_size)

# Initiate model
tf.keras.backend.clear_session()

if args.model_type == 'resnet':
    model, model_with_softmax = resnet_v1([32,32,3], 20)
    init_weight_copy = model.get_weights()
    model_with_softmax.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

elif args.model_type == 'small':
    model, model_with_softmax = cifar_small()
    init_weight_copy = model.get_weights()
    model_with_softmax.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

else:
    raise f"No model name {args.model_type}"


print(model.summary())

training_fn = {
    'db':double_back,
    'jac':JacReg,
    'fob':FobReg,
    'myfob1':MyFobReg1,
    'myfob2':MyFobReg2,
    'dbmyfob1':dbMyFobReg1,
    'dbmyfob1_v2':dbMyFobReg1_v2,
    'dbmyfob1_v3':dbMyFobReg1_v3,
    'dbfob':dbFobReg
}


epochs = args.epochs


x_in = tf.Variable(1.)

@tf.function()
def train_step(x_batch, y_batch, optimizer):
    x_batch = x_in * x_batch
    grads, loss, grad_fx = training_fn[args.reg](model_with_softmax, model, x_batch, y_batch, args.alpha)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

if args.reg == 'NI':
    loss_hist_NI = {
        'train':[],
        'val':[],
        'acc':[],
        'final_val':None,
        'final_acc':None

    }

    lr = 0.001
    model.set_weights(init_weight_copy)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for i in range(epochs):
        batches = 0
        for x_batch, y_batch in train_gen:
#             print(x_batch.shape)
            x_batch = x_batch + args.alpha * np.random.normal(size=x_batch.shape)
#             print( np.random.normal(x_batch.shape).shape)
#             print(x_batch.shape)

            model_with_softmax.fit(x_batch, y_batch, verbose=0)

            batches += 1
            if batches > len(x_train_small) / batch_size:
                break

        result_train = model_with_softmax.evaluate_generator(train_gen)
        #result_val = model_with_softmax.evaluate_generator(val_gen)
        #loss_hist_NI['train'].append(result_train[0])
        #loss_hist_NI['val'].append(result_val[0])
        #loss_hist_NI['acc'].append(result_val[1])
        print(result_train)
    result_val = model_with_softmax.evaluate_generator(val_gen)

    loss_hist_NI['final_val'] = result_val[0]
    loss_hist_NI['final_acc'] = result_val[1]

    with open(save_file_name, 'wb') as f:
        pickle.dump(loss_hist_NI, f)
else:
    loss_hist = {
        'train':[],
        'val':[],
        'acc':[],
        'final_val':None,
        'final_acc':None

    }

    lr = 0.001
    model.set_weights(init_weight_copy)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #print('fuck')

    for i in range(epochs):
        batches = 0
        for x_batch, y_batch in train_gen:
            #if args.add_noise:
            #    x_batch = x_batch + args.alpha_noise * np.random.normal(size=x_batch.shape)
            train_step(x_batch, y_batch, optimizer)
            x_batch = x_in * x_batch


            batches += 1
            if batches > len(x_train_small) / batch_size:
                break

        result_train = model_with_softmax.evaluate_generator(train_gen)
        #result_val = model_with_softmax.evaluate_generator(val_gen)
        #loss_hist['train'].append(result_train[0])
        #loss_hist['val'].append(result_val[0])
        #loss_hist['acc'].append(result_val[1])
        print(result_train)

    result_val = model_with_softmax.evaluate_generator(val_gen)
    loss_hist['final_val'] = result_val[0]
    loss_hist['final_acc'] = result_val[1]

    with open(save_file_name, 'wb') as f:
        pickle.dump(loss_hist, f)


model_with_softmax.save(save_model_name)
