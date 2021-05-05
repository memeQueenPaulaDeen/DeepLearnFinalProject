import os
import copy 
import numpy as np
from pickle import load, dump
import matplotlib.pyplot as plt

import keras as k 
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import *
from keras.regularizers import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator

from model_arch import *

def show_history(history):
    plt.figure(1)
    # plot loss
    plt.title('Cross Entropy Loss')
    plt.plot(history['loss'], color='blue', label='train')
    plt.plot(history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.figure(2)
    plt.title('Classification Accuracy')
    plt.plot(history['accuracy'], color='blue', label='train')#accuracy
    plt.plot(history['val_accuracy'], color='orange', label='test')#val_accuracy

    plt.show()
    print('Final Accuracy: ', history['val_accuracy'][-1])


def run_model(base_dir_out, in_files, run_params, seg_params):
	########## Program Variables ##########
	num_epochs, batch_size, optimizer, num_classes = run_params
	run_seg_model,seg_version, run_cost_model, _loss, params = seg_params
	file_x_seg, file_y_seg, file_x_cost, file_y_cost = in_files

	input_shape = (256, 256, 3)

	########### Generating Segmentation Model #########
	seg_model = None
	if run_seg_model:
		if seg_version == 'vgg-unet':
			print('Running VGG-Unet')
			seg_model = gen_VGG_unet_model(input_shape, num_classes, params)
		else:
			print('Running Unet')
			seg_model = gen_unet_model(input_shape,  num_classes, params)

		plot_model(seg_model, to_file=base_dir_out+'/seg_model.png', show_shapes=True)
		seg_model.compile(optimizer = optimizer, loss = _loss, metrics = ['accuracy'])

		es= k.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=7)
		cbs = [es]

		train_x_seg = np.load('./train/npy/'+file_x_seg+'.npy')
		train_y_seg = np.load('./train/npy/'+file_y_seg+'.npy')

		print('')
		print('')
		print('Len: X - {}, Y - {}'.format(len(train_x_seg), len(train_y_seg)))
		print('Shape: X - {}, Y - {}'.format(train_x_seg.shape, train_y_seg.shape))
		print('Max: X - {}, Y - {}'.format(np.max(train_x_seg), np.max(train_y_seg)))
		print('')

		history = seg_model.fit(train_x_seg, train_y_seg, \
				epochs = num_epochs, \
				batch_size = batch_size, 
				validation_split = 0.2, 
				verbose = 1, 
				callbacks = cbs)
                
		show_history(history.history)

		seg_model.save(base_dir_out+'/seg_model.h5')
		dump(history.history, open(base_dir_out+'/seg_history.pkl', 'wb'))
	else:
		seg_model = load_model(base_dir_out + '/seg_model.h5')
		history = load(open(base_dir_out+'/seg_history.pkl', 'rb'))
		show_history(history)

    ########### Generating and Training Model #########
	if run_cost_model:
		cost_model = gen_cost_model(seg_model, input_shape, num_classes)
		plot_model(cost_model, to_file=base_dir_out+'/cost_model.png', show_shapes=True)
		cost_model.compile(optimizer = optimizer, loss = 'mse', metrics = ['accuracy', 'mean_squared_error'])

		es2= k.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=7)
		cbs2 = [es2]

		train_x_cost = np.load('./train/npy/'+file_x_cost+'.npy')
		train_y_cost = np.load('./train/npy/'+file_y_cost+'.npy')

		print('')
		print('')
		print('Len: X - {}, Y - {}'.format(len(train_x_cost), len(train_y_cost)))
		print('Shape: X - {}, Y - {}'.format(train_x_cost.shape, train_y_cost.shape))
		print('Max: X - {}, Y - {}'.format(np.max(train_x_cost), np.max(train_y_cost)))
		print('')


		history = cost_model.fit(train_x_cost, train_y_cost, \
					epochs = num_epochs, \
					batch_size = batch_size, 
					validation_split = 0.2, 
					verbose = 1, 
					callbacks = cbs2)

		show_history(full_hist)


		cost_model.save(base_dir_out+'/cost_model.h5')
		dump(full_hist, open(base_dir_out+'/cost_history.pkl', 'wb'))


###########################################
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

base_dir_out ='./output/vgg_nocat_nobatch_relu_relu' 

file_x_seg = 'x_wbnxnyanc'
file_y_seg = 'y_wbnxnyanc'
file_x_cost = None
file_y_cost = None

in_files = (file_x_seg, file_y_seg, file_x_cost, file_y_cost)

num_epochs = 20
batch_size = 8
optimizer = Adam(lr=1e-4)
num_classes = 10

run_seg_model = True
run_cost_model = False
seg_version = 'vgg-unet' #'unet'
loss = 'mse'#'categorical_crossentropy'

cat = False
batch = False
act_type = 'relu'
out_type = 'relu'

run_params = (num_epochs, batch_size, optimizer, num_classes)
model_params = (cat, batch, act_type, out_type)
seg_params = (run_seg_model, seg_version, run_cost_model, loss, model_params)
run_model(base_dir_out, in_files, run_params, seg_params)
