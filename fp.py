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


def getColorForSegMap( mask: np.array) -> np.array:
    # in BGR for open cv lib Human readable

    mask = np.argmax(mask,axis=2)
    cmapFullClass = {
        'Background': (0, 0, 0),  # black
        'Building-flooded': (242, 23, 140),  # purple
        'Building-non-flooded': (27, 104, 239),  # orange
        'Road-flooded': (255, 255, 255),  # white
        'Road-non-flooded': (48, 84, 111),  # brown
        'Water': (242, 227, 23),  # light blue
        'Tree': (23, 242, 234),  # yellow
        'Vehicle': (125, 124, 125),  # grey
        'Pool': (202, 70, 13),  # dark blue
        'Grass': (0, 255, 0),  # green
    }

    # Translate encodeing to human readable
    m2Class = {
        0: 'Background',
        1: 'Building-flooded',
        2: 'Building-non-flooded',
        3: 'Road-flooded',
        4: 'Road-non-flooded',
        5: 'Water',
        6: 'Tree',
        7: 'Vehicle',
        8: 'Pool',
        9: 'Grass'
    }

    result = np.zeros((mask.shape[0], mask.shape[1], 3))

    for m in m2Class.keys():
        result[mask == m] = cmapFullClass[m2Class[m]]

    return result.astype('uint8')

class USHelp():

    def __init__(self,imgFolderPath):

        self.fpath = imgFolderPath

        # for img in os.listdir(self.fpath):
        #     self.manuelEst(img)

    def getGen(self, batch_size, inputShape = (256,256,3)):
        tg = k.preprocessing.image.ImageDataGenerator(validation_split=.2)

        XDir = os.path.join(self.fpath,'..')
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
            tg1 = k.preprocessing.image.ImageDataGenerator(validation_split=.2, channel_shift_range=.3,
                                                           brightness_range=(.7, 1))

            trainGen = tg1.flow_from_directory(XDir,
                                               target_size=inputShape[0:-1],
                                               batch_size=batch_size,
                                               color_mode="rgb",
                                               class_mode='input',
                                               subset='training', )
            # shuffle=False)

            files = trainGen.filenames
            names = [f.split(os.sep)[1] for f in files]
            while True:

                x = trainGen.__next__()[0]
                y = []
                fnames = names[trainGen.batch_index * batch_size - batch_size: trainGen.batch_index * batch_size]

                for xi in fnames:
                    y.append([self.manuelEst(xi)])
                y = np.concatenate(y)

                # for i in range(len(x)):
                #     xi = x[i]
                #     yi = y[i]
                #
                #     xi, yi = flip(xi, yi, )
                #     xi, yi = zoom(xi, yi, .2)
                #     xi, yi = shift(xi, yi, .3)
                #     xi, yi = rotate(xi, yi, 5)
                #     xi, yi = noise(xi, yi, .05)
                #
                #     x[i] = xi
                #     y[i] = yi

                # if categorical
                # y = (np.arange(y.max()+1) == y[...,None]).astype(int)

                yield x, y
                trainGen.reset()

        def generator_val():
            tg1 = k.preprocessing.image.ImageDataGenerator(validation_split=.2)

            valGen = tg1.flow_from_directory(XDir,
                                             target_size=inputShape[0:-1],
                                             batch_size=batch_size,
                                             color_mode="rgb",
                                             class_mode='input',
                                             subset='validation', )
            # shuffle=False)

            files = valGen.filenames
            names = [f.split(os.sep)[1] for f in files]
            while True:

                x = valGen.__next__()[0]
                y = []
                fnames = names[valGen.batch_index * batch_size - batch_size: valGen.batch_index * batch_size]

                for xi in fnames:
                    y.append([self.manuelEst(xi)])
                y = np.concatenate(y)

                # if categorical
                # y = (np.arange(y.max() + 1) == y[..., None]).astype(int)

                yield x, y
                valGen.reset()

        return generator_train(), generator_val(), trainItrPerEpoch, valItrPerEpoch

    def manuelEst(self,img):


        img = cv.imread(os.path.join(self.fpath, img))
        img = cv.resize(img, (256, 256))

        K = 5
        attempts = 10
        vectorized = img.reshape((-1, 3))
        vectorized = np.float32(vectorized)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

        center = np.uint8(center)

        res = center[label.flatten()]
        result_image = res.reshape((img.shape))

        new_label = np.zeros((img.shape[0],img.shape[1]))

        centSortedGreen = sorted(center, key=lambda tup: tup[1], reverse=True)
        minIdx = np.min(np.where(np.ptp(centSortedGreen,axis=1) == np.ptp(centSortedGreen,axis=1).min()))

        # new_label[(result_image==centSortedGreen[minIdx])[:,:,1]] = 0
        # centSortedGreen = np.delete(centSortedGreen,minIdx)

        idx = 0
        for trip in centSortedGreen:
            new_label[(result_image ==trip)[:,:,1]] = idx
            idx = idx +1




        # hmy = label.reshape((img.shape[0],img.shape[1])) / label.max() * 255
        # hmy = cv.applyColorMap(hmy.astype('uint8'), cv.COLORMAP_HOT)
        #
        # hmR = new_label.reshape((img.shape[0], img.shape[1])) / label.max() * 255
        # hmR = cv.applyColorMap(hmR.astype('uint8'), cv.COLORMAP_HOT)

        # cv.imshow('origin',img)
        # cv.imshow('seg',result_image)
        # cv.imshow('map',hmy)
        # cv.imshow('Rectified map',hmR)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return  new_label/ new_label.max()


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



def getGen(XDir,yDir,batchSize,inputShape,categorical = False):
    tg = k.preprocessing.image.ImageDataGenerator(validation_split=.2)
    trainSet = tg.flow_from_directory(XDir,
                                      target_size=inputShape[0:-1],
                                      batch_size=batchSize,
                                      color_mode="rgb",
                                      class_mode='categorical',
                                      subset='training',
                                      shuffle=False)
    valSet = tg.flow_from_directory(XDir,
                                    target_size=inputShape[0:-1],
                                    batch_size=batchSize,
                                    color_mode="rgb",
                                    class_mode='categorical',
                                    subset='validation',
                                    shuffle=False)


    augLoops = 1
    numAugs = 6
    trainItrPerEpoch = (int(len(trainSet.classes) / trainSet.batch_size)-1) * numAugs * augLoops
    valItrPerEpoch = int(len(valSet.classes) / valSet.batch_size) -1

    def generator_train():
        tg1 = k.preprocessing.image.ImageDataGenerator(validation_split=.2,channel_shift_range=.3,brightness_range=(.7,1))

        trainGen = tg1.flow_from_directory(XDir,
                                           target_size=inputShape[0:-1],
                                           batch_size=batchSize,
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
                y.append( [np.load(fname)])
            y = np.concatenate(y)

            augedx = []
            augedy = []
            for i in range(len(x)):
                xi = x[i]
                yi = y[i]

                for i in range(augLoops):
                    xi1, yi1 = flip(xi, yi,)
                    xi2, yi2 = zoom(xi, yi,.35+i/20)
                    xi3, yi3 = shift(xi,yi,.2+i/20)
                    xi4, yi4 = rotate(xi,yi,5+i)
                    xi5, yi5 = noise(xi,yi,.05+i/40)

                    augedx.append(xi), augedx.append(xi1), augedx.append(xi2), augedx.append(xi3), augedx.append(xi4), augedx.append(xi5)
                    augedy.append(yi), augedy.append(yi1), augedy.append(yi2), augedy.append(yi3), augedy.append(yi4), augedy.append(yi5)

            augedx = augedx[0:len(augedx) - len(augedx) % batchSize]
            augedy = augedy[0:len(augedy) - len(augedy) % batchSize]

            c = list(zip(augedx, augedy))
            random.shuffle(c)
            augedx, augedy = zip(*c)



            while len(augedx) > 0:

                nextX = augedx[0:batchSize]
                augedx = augedx[batchSize:]

                nextY = augedy[0:batchSize]
                augedy = augedy[batchSize:]

                x = np.concatenate([nextX])
                y = np.concatenate([nextY])

                if categorical:
                    y = (np.arange(10) == y[...,None]).astype(int)

                    yield x,y.astype('float32')

            if trainGen.batch_index*batchSize >= len(trainSet.classes)-batchSize:
                trainGen.reset()
                del x, y, augedx, augedy, nextY, nextX,c


    def generator_val():
        tg1 = k.preprocessing.image.ImageDataGenerator(validation_split=.2)

        valGen = tg1.flow_from_directory(XDir,
                                         target_size=inputShape[0:-1],
                                         batch_size=batchSize,
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
                y.append( [np.load(fname)])
            y = np.concatenate(y)

            if categorical:
                y = (np.arange(10) == y[..., None]).astype(int)

            yield x,y.astype('float32')

            if valGen.batch_index*batchSize >= len(valSet.classes) -batchSize:
                valGen.reset()
                del x, y

    # idx = 0
    # for x,y in generator_train():
    #
    #     bidx = 0
    #     for xi in x:
    #         cv.imshow('x',xi.astype('uint8'))
    #         cv.imshow('y',getColorForSegMap(y[bidx]))
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()
    #         bidx = bidx + 1
    #         idx = idx + 1
    #         print(idx)

    return generator_train(), generator_val(), trainItrPerEpoch, valItrPerEpoch


def activation(x, _type):
    if _type.lower() == 'leaky':
        return LeakyReLU(.1)(x)
    elif _type.lower() == 'relu':
        return k.layers.ReLU()(x)
    elif _type.lower() == 'max_relu':
        return k.layers.ReLU(max_value=1.0)(x)
    elif _type.lower == 'softmax':
        return k.layers.Softmax()(x)

def gen_VGG_unet_model(input_shape, num_classes, params):
    cat, batch, _type, out_type = params

    #https://www.kaggle.com/basu369victor/transferlearning-and-unet-to-segment-rocks-on-moon

    vgg = k.applications.vgg16.VGG16(include_top=False, weights= 'imagenet',input_shape=input_shape)

    for layer in vgg.layers:
        layer.trainable = False

    un = Conv2DTranspose(256,(9,9),strides=(2,2),padding='same')(vgg.output)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    concat_1 = concatenate([un, vgg.get_layer("block5_conv3").output])

    un = Conv2D(512,(3,3),strides=(1,1),padding='same')(concat_1)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    un = Conv2DTranspose(512, (3, 3), strides=(2, 2),padding='same')(un)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    concat_2 = concatenate([un, vgg.get_layer("block4_conv3").output])

    un = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(concat_2)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    un = Conv2DTranspose(512, (3, 3), strides=(2, 2),padding='same')(un)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    concat_3 = concatenate([un, vgg.get_layer("block3_conv3").output])

    un = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(concat_3)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    un = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(un)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    concat_4 = concatenate([un, vgg.get_layer("block2_conv2").output])

    un = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(concat_4)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    un = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(un)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    concat_5 = concatenate([un, vgg.get_layer("block1_conv2").output])

    un = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(concat_5)
    un = activation(un, _type)
    if batch:
        un = BatchNormalization()(un)

    if cat:
      # reconstruct segmentation map -> try one hot encoding style (num layers = num classes)
      un = Conv2D(num_classes,(3,3),strides=(1, 1),padding='same')(un)
    else:
      # reconstruct segmentation map as 1 layer labeled image
      un = Conv2D(1,(3,3),strides=(1, 1),padding='same')(un)
      # weighting = tf.Tensor([5000, 5000, 5000, 5000, 1, 5000, 500, 5000, 5000, 50], 'float32')
      # un = k.layers.Lambda(lambda x: tf.matmul(x, tf.expand_dims(weighting, axis=1)))(un)

    un = activation(un, out_type)

    un = Model(inputs= vgg.input, outputs= un)

    return un



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

    DSb3 = Conv2D(512, (1, 1), strides=(4, 4), padding='same')(vgg.get_layer("block3_conv3").output)
    DSb4 =  Conv2D(512,(1,1),strides=(2,2),padding='same')(vgg.get_layer("block4_conv3").output)
    concat_1 = concatenate([un, vgg.get_layer("block5_conv3").output])#,DSb4,DSb3])

    un = Conv2D(512,(3,3),strides=(1,1),padding='same')(concat_1)
    un = LeakyReLU(.1)(un)
    #un = BatchNormalization()(un)

    un = Conv2DTranspose(512, (3, 3), strides=(2, 2),padding='same')(un)
    un = LeakyReLU(.1)(un)
    #un = BatchNormalization()(un)

    DSb3 = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(vgg.get_layer("block3_conv3").output)
    UsB5 = Conv2DTranspose(64, (3, 3), strides=(2, 2),padding='same')(vgg.get_layer("block5_conv3").output)
    concat_2 = concatenate([un, vgg.get_layer("block4_conv3").output])#,DSb3,UsB5])

    un = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(concat_2)
    un = LeakyReLU(0.1)(un)
    #un = BatchNormalization()(un)

    un = Conv2DTranspose(512, (3, 3), strides=(2, 2),padding='same')(un)
    un = LeakyReLU(0.1)(un)
    #un = BatchNormalization()(un)

    DSb2 = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(vgg.get_layer("block2_conv2").output)
    UsB5 = Conv2DTranspose(64, (5, 5), strides=(4, 4),padding='same')(vgg.get_layer("block5_conv3").output)
    UsB4 = Conv2DTranspose(64, (3, 3), strides=(2, 2),padding='same')(vgg.get_layer("block4_conv3").output)
    concat_3 = concatenate([un, vgg.get_layer("block3_conv3").output])

    un = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(concat_3)
    un = LeakyReLU(0.1)(un)
    #un = BatchNormalization()(un)

    un = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(un)
    un = LeakyReLU(0.1)(un)
    #un = BatchNormalization()(un)

    concat_4 = concatenate([un, vgg.get_layer("block2_conv2").output])

    un = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(concat_4)
    un = LeakyReLU(0.1)(un)
    #un = BatchNormalization()(un)

    un = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(un)
    un = LeakyReLU(0.1)(un)
    #un = BatchNormalization()(un)

    concat_5 = concatenate([un, vgg.get_layer("block1_conv2").output])

    un = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(concat_5)
    un = LeakyReLU(0.1)(un)
    #un = BatchNormalization()(un)



    un = Conv2D(1, (3,3), strides=(1, 1), padding='same',activation='softmax')(un)

    ##ramp max activation
    #un = Conv2D(1, (3,3), strides=(1, 1), padding='same',activation=rampMax)(un)
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

def rampMax(x):
    return k.backend.relu(x,max_value=1)

def run_model(run_params, model_params,generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch,categorical):
    ########## Program Variables ##########
    num_epochs, batch_size, optimizer = run_params
    input_shape, num_classes = model_params

    ########### Generating and Training Model #########
    #model = gen_model(input_shape, num_classes)

    #categorical, Do batch Norm, _type, out_type
    p = (categorical,False,'relu','relu')
    model = gen_VGG_unet_model(input_shape,num_classes,p)

    if not categorical:
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    k.utils.plot_model(model, 'mvgg.png', show_shapes=True)

    es= k.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=7)
    rlronp = k.callbacks.ReduceLROnPlateau(monitor='loss',factor=.5,patience=3)
    cbs = [es,rlronp]

    # XTrainDir = os.path.join('DD','X_Train')
    # YTrainDir = os.path.join('DD','Y_Train')

    # us = USHelp(os.path.join('Unlabeled','image'))
    #
    # generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch = us.getGen(batch_size)

    history = model.fit(generator_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        steps_per_epoch=trainItrPerEpoch,
                        validation_steps=valItrPerEpoch,
                        validation_data=generator_val,
                        shuffle=True,
                        callbacks=cbs)

    #model = k.models.load_model(os.path.join('models','m13'))



    # for layer in model.layers:
    #     if layer.name in ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']:
    #         layer.trainable = False
    #     # elif layer.name in ['conv2d_transpose','conv2d','batch_normalization','conv2d_transpose_1','conv2d_1','batch_normalization_1','conv2d_transpose_2','conv2d_2','batch_normalization_2']:
    #     #     layer.trainable = False
    #     else:
    #         layer.trainable = True

    # XTrainDir = 'X_Train_256'
    # YTrainDir = 'Y_TrainNormBlur_256'

    # generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch = getGen(XTrainDir, YTrainDir, batch_size,input_shape)
    #
    # #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy',"mean_squared_error"])
    #
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

    num_epochs = 20
    batchSize = 8
    optimizer = Adam(lr=1e-4)

    Weighting = {
        'Obstacle': 10000,
        'Tree': 300,
        'Grass': 50,
        'Road-non-flooded': 1
    }

    categorical = False

    pwd = os.path.dirname(os.path.abspath(sys.argv[0]))

    if categorical:
        XTrainDir = 'X_Train_OG_256'
        YTrainDir = 'Y_Train_OG_256'
    else:
        XTrainDir = 'X_Train_256'
        YTrainDir = 'Y_TrainNormBlur_256'


    input_shape = (256, 256, 3)
    num_classes = 10#len(Weighting)

    generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch = getGen(XTrainDir, YTrainDir, batchSize, input_shape,categorical=categorical)

    run_params = (num_epochs, batchSize, optimizer)
    model_params = (input_shape, num_classes)

    history, model = run_model(run_params, model_params,generator_train, generator_val, trainItrPerEpoch, valItrPerEpoch,categorical)

    model.save(os.path.join(pwd,'models','m14'))
    dump(history.history, open(pwd + 'history.pkl', 'wb'))
