
from tensorflow import keras as k
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose, BatchNormalization, concatenate, Conv2D

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class VGG_UNET():

    def __init__(self,input_shape,num_classes,batchSize,isCategorical,doBatchNorm,hiddenDecoderActivation,outputActivation):
        self.input_shape=input_shape
        self.num_classes = num_classes
        self.isCategorical = isCategorical
        self.doBatchNorm = doBatchNorm
        self.hiddenDecoderActivation=hiddenDecoderActivation
        self.outputActivation = outputActivation
        self.batchSize = batchSize
        # self.generator = generator

        self.model = self.gen_model()

    def activation(self, x, _type):
        if _type.lower() == 'leaky':
            return LeakyReLU(.1)(x)
        elif _type.lower() == 'relu':
            return k.layers.ReLU()(x)
        elif _type.lower() == 'max_relu':
            return k.layers.ReLU(max_value=1.0)(x)
        elif _type.lower() == 'softmax':
            return k.layers.Softmax()(x)
        else:
            assert False, "specified activation does not exist or was not implemented"

    def gen_model(self):

        # https://www.kaggle.com/basu369victor/transferlearning-and-unet-to-segment-rocks-on-moon

        vgg = k.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)

        for layer in vgg.layers:
            layer.trainable = False

        un = Conv2DTranspose(256, (9, 9), strides=(2, 2), padding='same')(vgg.output)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        concat_1 = concatenate([un, vgg.get_layer("block5_conv3").output])

        un = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(concat_1)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        un = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(un)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        concat_2 = concatenate([un, vgg.get_layer("block4_conv3").output])

        un = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(concat_2)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        un = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(un)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        concat_3 = concatenate([un, vgg.get_layer("block3_conv3").output])

        un = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(concat_3)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        un = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(un)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        concat_4 = concatenate([un, vgg.get_layer("block2_conv2").output])

        un = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(concat_4)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        un = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(un)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        concat_5 = concatenate([un, vgg.get_layer("block1_conv2").output])

        un = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(concat_5)
        un = self.activation(un, self.hiddenDecoderActivation)
        if self.doBatchNorm:
            un = BatchNormalization()(un)

        if self.isCategorical:
            # reconstruct segmentation map -> try one hot encoding style (num layers = num classes)
            un = Conv2D(self.num_classes, (3, 3), strides=(1, 1), padding='same')(un)
        else:
            # reconstruct segmentation map as 1 layer labeled image

            un = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(un)
            # weighting = tf.Tensor([5000, 5000, 5000, 5000, 1, 5000, 500, 5000, 5000, 50], 'float32')
            # un = k.layers.Lambda(lambda x: tf.matmul(x, tf.expand_dims(weighting, axis=1)))(un)

        un = self.activation(un, self.outputActivation)

        un = Model(inputs=vgg.input, outputs=un)

        return un