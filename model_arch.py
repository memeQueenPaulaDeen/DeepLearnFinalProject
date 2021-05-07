from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import keras as k 

def activation(x, _type):
    if _type.lower() == 'leaky':
        return LeakyReLU(.1)(x)
    elif _type.lower() == 'relu':
        return ReLU()(x)
    elif _type.lower() == 'max-relu':
        return ReLU(max_value=1.0)(x)
    #elif _type.lower == 'softmax':
        #return Activation(k.activations.softmax)(x)

def gen_cost_model(seg_model, input_shape, num_classes):

    num = 0
    for layer in seg_model.layers:
        layer.trainable = True
        layer._name = 'layer_{}'.format(num)
        num += 1

    # Cost Map extension
    conv11 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='cost_1')(seg_model.output)
    conv11 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='cost_2')(conv11)
    conv11 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='cost_3')(conv11)
    conv11 = Conv2D(1, 3, padding = 'same', kernel_initializer = 'he_normal', name='cost_4')(conv11)  
    
    conv11 = activation(conv11, 'max-relu')

    cost_model = Model(inputs= seg_model.input, outputs= conv11)
    return cost_model


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
		if out_type == 'softmax':
			un = Conv2D(num_classes,(3,3),strides=(1, 1),padding='same', activation='softmax')(un)
		else:
			un = Conv2D(num_classes,(3,3),strides=(1, 1),padding='same')(un)     
	else:
		# reconstruct segmentation map as 1 layer labeled image 
		if out_type  == 'softmax':
			un = Conv2D(1,(3,3),strides=(1, 1),padding='same', activation='softmax')(un)
		else:
			un = Conv2D(1,(3,3),strides=(1, 1),padding='same')(un)
	
	if out_type != 'softmax':
		un = activation(un, out_type)

	un = Model(inputs= vgg.input, outputs= un)

	return un



def gen_unet_model(input_shape, num_classes, params):
	cat, batch, _type, out_type = params

	#U-Net implementation 
	#https://github.com/zhixuhao/unet/blob/master/model.py
	img_input = Input(input_shape) #
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_input)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

	if cat:
		if out_type == 'softmax':
			conv10 = Conv2D(10, 3, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		else:
			conv10 = Conv2D(10, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	else:
		conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

	model = Model(inputs = img_input, outputs = conv10)
	return model


