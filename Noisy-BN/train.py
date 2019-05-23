import argparse
import os

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', default='fashion')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.)
    parser.add_argument('--p', dest='p', type=float, default=0.25)
    parser.add_argument('--suffix', dest='suffix')
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--gpu', dest='gpu', type=int, default=0)
    parser.add_argument('--train_size', dest='train_size', type=int, default=None)
    parser.add_argument('--small', dest='save', const=True, 
                        action='store_const', default=False)
    parser.add_argument('--no_save', dest='save', const=True, 
                        action='store_const', default=False)

    return parser.parse_args()

args = parser()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu - 1)
import pickle


#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from layers.noisy_bn import NoisyBatchNormalization



def create_folder(name):
    '''
    Create a folder. if exist, pass.
    '''
    try:
        os.makedirs(name)
    except FileExistsError:
        print('Folder {} was created'.format(name))


def main(dataset, alpha, p, suffix, epochs, train_size, **kwargs):

    if dataset == 'fashion':

        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)

        train_images, train_labels = cut_data(train_size, train_images, train_labels)

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True,
            horizontal_flip=True,)

        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True)

        train_datagen.fit(train_images)
        # val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

        val_datagen.mean = train_datagen.mean
        val_datagen.std = train_datagen.std

        train_gen = train_datagen.flow(train_images, to_categorical(train_labels), batch_size=100)
        val_gen = val_datagen.flow(test_images, to_categorical(test_labels), shuffle=False, batch_size=100)

        if kwargs.small:
            from models.fashion_mnist import get_small_model
            model = get_small_model(alpha=alpha, p=p)
        else:
            from models.fashion_mnist import get_model
            model = get_model(alpha=alpha, p=p)

    if dataset == 'cifar':
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

        train_images, train_labels = cut_data(train_size, train_images, train_labels)

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None)

        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            featurewise_std_normalization=True,) # divide inputs by std of the dataset

        train_datagen.fit(train_images)
        val_datagen.mean = train_datagen.mean
        val_datagen.std = train_datagen.std

        train_gen = train_datagen.flow(train_images, to_categorical(train_labels), batch_size=32)
        val_gen = val_datagen.flow(test_images, to_categorical(test_labels), shuffle=False, batch_size=32)

        if kwargs.small:
            from models.CIFAR10 import get_small_model
            model = get_small_model(alpha=alpha, p=p)
        else:
            from models.CIFAR10 import WideResidualNetwork
            model = WideResidualNetwork(depth=16, alpha=alpha, p=p)


    print(f'Running dataset={dataset} alpha={alpha}, p={p}, round={suffix} in {epochs} epochs')



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    create_folder(f'models/files/{dataset}_{alpha}_{p}_{suffix}')
    create_folder(f'models/files/{dataset}_{alpha}_{p}_{suffix}/logs')

    lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=f'models/files/{dataset}_{alpha}_{p}_{suffix}/model', monitor='val_accuracy')
    tf_board = keras.callbacks.TensorBoard(log_dir=f"models/files/{dataset}_{alpha}_{p}_{suffix}/logs", histogram_freq=1)
    callbacks = [lr_reduce, tf_board]

    if not kwargs.no_save:
        callbacks.append(checkpoint)

    tmp = model.fit_generator(train_gen,
                        epochs=epochs,
                        validation_data=val_gen,
                        verbose=1,
                        callbacks=callbacks,
                        workers=4)

    with open(f"models/files/{dataset}_{alpha}_{p}_{suffix}/history.pickle", 'wb') as f:
        pickle.dump(tmp.history, f)

def cut_data(train_size, train_images, train_labels):
    np.random.seed(0)
    if train_size is not None:
        chosen_idx = np.random.permutation(len(train_images))[:train_size]
        train_images = train_images[chosen_idx]
        train_labels = train_labels[chosen_idx]
    return train_images, train_labels

if __name__ == '__main__':
    main(**vars(args))
