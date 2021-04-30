import os
import random
import sys

import numpy as np
from pickle import dump
from keras.preprocessing.image import load_img
import keras as k
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Add, Conv2DTranspose
from keras.layers import Activation, Dropout, MaxPooling2D, LeakyReLU ,AveragePooling2D, Flatten, concatenate, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, Adagrad, RMSprop, schedules
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv

import tensorflow as tf
import numpy as np
from pickle import load, dump

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def shift(x,y,ratio= 0.0):
    #https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
    assert ratio < 1 and ratio > 0, 'Value should be less than 1 and greater than 0'

    ratioh = random.uniform(-ratio, ratio)
    ratiov = random.uniform(-ratio, ratio)

    h, w = x.shape[:2]

    moveh = w * ratioh
    movev = w * ratiov

    if ratioh > 0:
        x = x[:, :int(w - moveh), :]
    if ratioh < 0:
        x = x[:, int(-1 * moveh):, :]

    if ratiov > 0:
        x = x[:int(h - movev), :, :]
    if ratiov < 0:
        x = x[int(-1 * movev):, :, :]

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)

    if ratioh > 0:
        y = y[:, :int(w - moveh), :]
    if ratioh < 0:
        y = y[:, int(-1 * moveh):, :]

    if ratiov > 0:
        y = y[:int(h - movev), :, :]
    if ratiov < 0:
        y = y[int(-1 * movev):, :, :]



    x = fill(x, h, w)
    y = fill(y, h, w)

    return x, np.squeeze(y)

def rotate(x,y,angle):

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)

    angle = int(random.uniform(-angle, angle))
    h, w = x.shape[:2]

    A = cv.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    x = cv.warpAffine(x, A, (w, h))
    y = cv.warpAffine(y, A, (w, h))

    return x, np.squeeze(y)

def flip(x,y):
    hf = random.randint(0,1)
    vf = random.randint(0,1)

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)

    if hf == 1:
        x = cv.flip(x,1)
        y = cv.flip(y,1)

    if vf == 1:
        x = cv.flip(x,0)
        y = cv.flip(y,0)

    return x, np.squeeze(y)

def zoom(x, y, range):
    #https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
    assert range < 1 and range > 0, 'Value should be less than 1 and greater than 0'
    range = random.uniform(range, 1)

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)


    h, w = x.shape[:2]
    h_taken = int(range * h)
    w_taken = int(range * w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    x = x[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    y = y[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    x = fill(x, h, w)
    y = fill(y, h, w)

    return x, np.squeeze(y)

def fill(img, h, w):
    #https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
    img = cv.resize(img, (h, w), cv.INTER_CUBIC)
    return img

def noise(x,y,cut):
    c = int(255*cut)
    n1 = np.random.randint(0, 255, (256, 256, 3))
    n2 = np.random.randint(0, 255, (256, 256, 3))

    n1[n1> c] = 0
    n2[n2 < 255-c] = 0
    x = x + (n1+n2)
    x[x> 255] = 255
    return x, y



def getGen(XDir,yDir,batchSize,inputShape):
    tg = k.preprocessing.image.ImageDataGenerator(validation_split=.2)
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
        tg1 = k.preprocessing.image.ImageDataGenerator(validation_split=.2,channel_shift_range=.2,brightness_range=(.8,1))

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

            for i in range(len(x)):
                xi = x[i]
                yi = y[i]

                xi, yi = flip(xi, yi,)
                xi, yi = zoom(xi, yi,.05)
                xi, yi = shift(xi,yi,.2)
                xi, yi = rotate(xi,yi,5)
                xi, yi = noise(xi,yi,.05)

                x[i] = xi
                y[i] = yi


            #if categorical
            y = (np.arange(y.max()+1) == y[...,None]).astype(int)

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

            # if categorical
            y = (np.arange(y.max() + 1) == y[..., None]).astype(int)

            yield x,y
            valGen.reset()

    return generator_train(), generator_val(), trainItrPerEpoch, valItrPerEpoch


def genVGGBasedModel(input_shape):

    #https://www.kaggle.com/basu369victor/transferlearning-and-unet-to-segment-rocks-on-moon

    vgg = k.applications.vgg16.VGG16(include_top=False, weights= 'imagenet',input_shape=input_shape)

    # for layer in vgg.layers:
    #     if layer.name in ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']:
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True

    for layer in vgg.layers:
        layer.trainable = False

    un = Conv2DTranspose(256,(9,9),strides=(2,2),padding='same')(vgg.output)
    un = LeakyReLU(.1)(un)
    un = BatchNormalization()(un)

    concat_1 = concatenate([un, vgg.get_layer("block5_conv3").output])

    un = Conv2D(512,(3,3),strides=(1,1),padding='same')(concat_1)
    un = LeakyReLU(.1)(un)
    un = BatchNormalization()(un)

    un = Conv2DTranspose(512, (3, 3), strides=(2, 2),padding='same')(un)
    un = LeakyReLU(.1)(un)
    un = BatchNormalization()(un)

    concat_2 = concatenate([un, vgg.get_layer("block4_conv3").output])

    un = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(concat_2)
    un = LeakyReLU(0.1)(un)
    un = BatchNormalization()(un)

    un = Conv2DTranspose(512, (3, 3), strides=(2, 2),padding='same')(un)
    un = LeakyReLU(0.1)(un)
    un = BatchNormalization()(un)

    concat_3 = concatenate([un, vgg.get_layer("block3_conv3").output])

    un = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(concat_3)
    un = LeakyReLU(0.1)(un)
    un = BatchNormalization()(un)

    un = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(un)
    un = LeakyReLU(0.1)(un)
    un = BatchNormalization()(un)

    concat_4 = concatenate([un, vgg.get_layer("block2_conv2").output])

    un = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(concat_4)
    un = LeakyReLU(0.1)(un)
    un = BatchNormalization()(un)

    un = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(un)
    un = LeakyReLU(0.1)(un)
    un = BatchNormalization()(un)

    concat_5 = concatenate([un, vgg.get_layer("block1_conv2").output])

    un = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(concat_5)
    un = LeakyReLU(0.1)(un)
    un = BatchNormalization()(un)


    un = Conv2D(1, (3,3), strides=(1, 1), padding='same',activation='sigmoid')(un)
    # un = BatchNormalization()(un)

    un = Model(inputs= vgg.input, outputs= un)

    return un



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
    # conv11 = Conv2D(num_classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    # conv11 = Conv2D(num_classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    # conv11 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv11)

    model = k.Model(inputs=img_input,outputs=conv10)
    return model


def run_model(run_params, model_params,generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch):
    ########## Program Variables ##########
    num_epochs, batch_size, optimizer = run_params
    input_shape, num_classes = model_params

    ########### Generating and Training Model #########
    #model = gen_model(input_shape, num_classes)
    model = genVGGBasedModel(input_shape)

    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy',"mean_squared_error"])
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    k.utils.plot_model(model, 'mvgg.png', show_shapes=True)

    es= k.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=7)
    cbs = [es]

    history = model.fit(generator_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        steps_per_epoch=trainItrPerEpoch,
                        validation_steps=valItrPerEpoch,
                        validation_data=generator_val,
                        callbacks=cbs)

    for layer in model.layers:
        if layer.name in ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']:
            layer.trainable = False
        else:
            layer.trainable = True

    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # print('fine tune')
    # history = model.fit(generator_train,
    #                     epochs=num_epochs,
    #                     batch_size=batch_size,
    #                     steps_per_epoch=trainItrPerEpoch,
    #                     validation_steps=valItrPerEpoch,
    #                     validation_data=generator_val,
    #                     callbacks=cbs)

    return history, model


###########################################

if __name__ == '__main__':

    num_epochs = 100
    batch_size = 4
    optimizer = Adam(lr=1e-4)

    Weighting = {
        'Obstacle': 5000,
        'Tree': 300,
        'Grass': 50,
        'Road-non-flooded': 1
    }

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    XTrainDir = 'X_Train_256'
    YTrainDir = 'Y_TrainNorm_256'


    input_shape = (256, 256, 3)
    num_classes = len(Weighting)

    generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch = getGen(XTrainDir,YTrainDir,batch_size,input_shape)

    run_params = (num_epochs, batch_size, optimizer)
    model_params = (input_shape, num_classes)

    history, model = run_model(run_params, model_params,generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch)

    model.save(os.path.join(pwd,'models','m6'))
    dump(history.history, open(pwd + '/history.pkl', 'wb'))
