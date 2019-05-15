import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', default='fashion')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.)
    parser.add_argument('--p', dest='p', type=float, default=0.25)
    parser.add_argument('--suffix', dest='suffix')
    parser.add_argument('--epochs', dest='epochs', type=int)

    return parser.parse_args()


def main(dataset, alpha, p, suffix, epochs):

    if dataset == 'fashion':

        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization = True,
        #     rotation_range=10,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     shear_range=0.1,
        #     zoom_range=0.1,
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

        from models.fashion_mnist import get_model
        model = get_model(alpha=alpha, p=p)

    print(f'Running dataset={dataset} alpha={alpha}, p={p}, round={suffix} in {epochs} epochs')



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    create_folder(f'models/files/fashion_{alpha}_{p}_{suffix}')
    create_folder(f'models/files/fashion_{alpha}_{p}_{suffix}/logs')

    lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=f'models/files/fashion_{alpha}_{p}_{suffix}/model', monitor='val_accuracy')
    tf_board = keras.callbacks.TensorBoard(log_dir=f"models/files/fashion_{alpha}_{p}_{suffix}/logs", histogram_freq=1)
    callbacks = [lr_reduce, checkpoint, tf_board]

    tmp = model.fit_generator(train_gen,
                        epochs=epochs,
                        validation_data=val_gen,
                        verbose=1,
                        callbacks=callbacks)

    with open(f"models/files/fashion_{alpha}_{p}_{suffix}/history.pickle", 'wb') as f:
        pickle.dump(tmp.history, f)

if __name__ == '__main__':
    args = parser()
    main(**vars(args))
