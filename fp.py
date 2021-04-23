import os
import sys

import numpy as np
from pickle import dump
from keras.preprocessing.image import load_img
import keras as k
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Add
from keras.layers import Activation, Dropout, MaxPooling2D, AveragePooling2D, Flatten, concatenate, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, Adagrad, RMSprop, schedules
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
from pickle import load, dump

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



def getGen(XDir,yDir,batchSize,inputShape):
    tg = k.preprocessing.image.ImageDataGenerator(validation_split=.02)
    trainSet = tg.flow_from_directory(XDir,
                                      target_size=inputShape[0:-1],
                                      batch_size=batch_size,
                                      color_mode="rgb",
                                      class_mode='categorical',
                                      subset='training',
                                      shuffle=False)
    valSet = tg.flow_from_directory(XDir,
                                      target_size=inputShape[0:-1],
                                      batch_size=batch_size,
                                      color_mode="rgb",
                                      class_mode='categorical',
                                      subset='validation',
                                      shuffle=False)

    trainItrPerEpoch = int(len(trainSet.classes) / trainSet.batch_size)
    valItrPerEpoch = int(len(valSet.classes) / valSet.batch_size)

    def generator_train():
        tg1 = k.preprocessing.image.ImageDataGenerator(validation_split=.2,channel_shift_range=.4)

        trainGen = tg1.flow_from_directory(XDir,
                                          target_size=inputShape[0:-1],
                                          batch_size=batch_size,
                                          color_mode="rgb",
                                          class_mode='input',
                                          subset='training',
                                          shuffle=False)

        files = trainGen.filenames
        ynames = [os.path.join(yDir,f.split(os.sep)[-1].split('.')[0]+'.npy') for f in files]
        while True:

            x = trainGen.__next__()[0]
            y = []
            fnames = ynames[trainGen.batch_index*batchSize - batchSize: trainGen.batch_index*batchSize]
            for fname in fnames:
                y.append( [np.load(fname).transpose()])
            y = np.concatenate(y)

            yield x,y
            trainGen.reset()


    def generator_val():
        tg1 = k.preprocessing.image.ImageDataGenerator(validation_split=.2)

        valGen = tg1.flow_from_directory(XDir,
                                          target_size=inputShape[0:-1],
                                          batch_size=batch_size,
                                          color_mode="rgb",
                                          class_mode='input',
                                          subset='validation',
                                          shuffle=False)

        files = valGen.filenames
        ynames = [os.path.join(yDir,f.split(os.sep)[-1].split('.')[0]+'.npy') for f in files]
        while True:

            x = valGen.__next__()[0]
            y = []
            fnames = ynames[valGen.batch_index*batchSize - batchSize: valGen.batch_index*batchSize]
            for fname in fnames:
                y.append( [np.load(fname).transpose()])
            y = np.concatenate(y)

            yield x,y
            valGen.reset()

    return generator_train(), generator_val(), trainItrPerEpoch, valItrPerEpoch



def gen_model(input_shape, num_classes):
    # U-Net implementation
    # https://github.com/zhixuhao/unet/blob/master/model.py
    img_input = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(img_input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    # Cost Map extension
    conv11 = Conv2D(num_classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv11 = Conv2D(num_classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    conv11 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv11)

    model = k.Model(inputs=img_input,outputs=conv11)
    return model


def run_model(run_params, model_params,generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch):
    ########## Program Variables ##########
    num_epochs, batch_size, optimizer = run_params
    input_shape, num_classes = model_params

    ########### Generating and Training Model #########
    model = gen_model(input_shape, num_classes)

    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy',"mean_squared_error"])
    print(model.summary())
    k.utils.plot_model(model, 'm', show_shapes=True)

    history = model.fit(generator_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        steps_per_epoch=trainItrPerEpoch,
                        validation_steps=valItrPerEpoch,
                        validation_data=generator_val)

    return history, model


###########################################

if __name__ == '__main__':

    num_epochs = 100
    batch_size = 2
    optimizer = Adam(lr=1e-4)

    Weighting = {
        'Obstacle': 5000,
        'Tree': 300,
        'Grass': 50,
        'Road-non-flooded': 1
    }

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    XTrainDir = 'X_Train'
    YTrainDir = 'Y_TrainNormBlur'


    input_shape = (512, 512, 3)
    num_classes = len(Weighting)

    generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch = getGen(XTrainDir,YTrainDir,batch_size,input_shape)

    run_params = (num_epochs, batch_size, optimizer)
    model_params = (input_shape, num_classes)

    history, model = run_model(run_params, model_params,generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch)

    model.save(pwd + '/model.h5')
    dump(history.history, open(pwd + '/history.pkl', 'wb'))
