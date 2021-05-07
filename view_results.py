import os
import sys
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

def convert_mask(mask, mask_type=0):
    new_mask = np.zeros((mask.shape[0], mask.shape[1]))

    if mask_type == 0:
        for i in range(mask.shape[2]):
            new_mask[mask[:,:,i] == 1] = i*20
        return new_mask

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            max_val = -10000
            max_val_idx = 0
            for layer in range(mask.shape[2]):
                if mask[row, col, layer] > max_val:
                    max_val = mask[row, col, layer]
                    max_val_idx = layer
            new_mask[row, col] = max_val_idx*20

    return new_mask

seg_model_load = False

if seg_model_load:
	model_name = 'seg_model.h5'
	file_x = 'x_wbnxnyac'
	file_y = 'y_wbnxnyac'

else:
	model_name = 'cost_model.h5'
	file_x = 'x_wbbnxyanc'
	file_y = 'y_wbbnxyanc'
	
model = keras.models.load_model('./output/cost_vgg_nocat_batch_relu_maxrelu_nonormcaty_normcosty_trainable_blur20/'+model_name)
file_n = np.load('./train/npy/file_names.npy')
file_names = []
for file in file_n:
	file_names.append(file)
	file_names.append(file)
	file_names.append(file)
	file_names.append(file)
	file_names.append(file)
	file_names.append(file)

if sys.argv[1] == '0':

	train_x = np.load('./train/npy/'+file_x+'.npy')
	train_y = np.load('./train/npy/'+file_y+'.npy')

	max_val = len(train_x)
	
	while True:
		num = np.random.randint(max_val)
		a = int(input('Next: '))
		if a == -1:
			break
		print('Filename : ', file_names[num])
		
		_input = train_x[num]
		mask = train_y[num]

		_input = np.expand_dims(_input, axis=0)
		output = model.predict(_input, verbose=0)[0]

		mask_new = None
		output_new = None
		if seg_model_load:
			mask_new = convert_mask(mask, 0)
			output_new = convert_mask(output, 1)

		plt.figure(1)
		plt.imshow(_input[0])
		if seg_model_load:
			plt.figure(2)
			plt.imshow(output_new, cmap='Set1')
			plt.figure(3)
			plt.imshow(mask_new, cmap='Set1')
		else:
			plt.figure(2)
			plt.imshow(output[:,:, 0], cmap='hot')
			plt.figure(3)
			plt.imshow(mask, cmap='hot')
		plt.show()

		plt.close(1)
		plt.close(2)
		plt.close(3)
elif sys.argv[1] == '1':
	files = []
	for file in os.listdir('./test/image_001'):
		files.append('./test/image_001/'+file)

	while True:
		num = np.random.randint(len(files))	
		a = int(input('Next: '))
		if a == -1:
			break
		print('Filename: ', files[num])
		
		img = cv2.imread(files[num])
		if img.shape[0] > 256 and img.shape[1] > 256:
			cx = int(img.shape[0]/2)
			cy = int(img.shape[1]/2)
			img = img [cx-256:cx+256, cy-256:cy+256]

			img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)

		_input = np.expand_dims(img, axis=0)
		out = model.predict(_input, verbose=0)[0]
		out = convert_mask(out, 1)

		plt.figure(1)
		plt.imshow(img)
			
		plt.figure(2)
		plt.imshow(out)
		plt.show()
		
		plt.close(1)
		plt.close(2)
else:
	img = cv2.imread(sys.argv[1])
	print(img.shape)	
	if img.shape[0] > 256 and img.shape[1] > 256:
		cx = int(img.shape[0]/2)
		cy = int(img.shape[1]/2)
		img = img [cx-256:cx+256, cy-256:cy+256]
		print(img.shape)

		img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
		print(img.shape)

	_input = np.expand_dims(img, axis=0)
	out = model.predict(_input, verbose=0)[0]
	out = convert_mask(out, 1)

	plt.figure(1)
	plt.imshow(img)
		
	plt.figure(2)
	plt.imshow(out)
	plt.show()
	
	plt.close(1)
	plt.close(2)

