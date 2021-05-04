import os
import cv2
import keras.preprocessing.image as kpi
import numpy as np
import img_funcs as img

    
def save_names(in_dir, exten_type, out_dir):
    file_names = []
    exten_len = -1*(len(exten_type)+1)
    for file in os.listdir(in_dir):
        file_names.append(file[:exten_len])
    print('Num Files: ', len(file_names))
    np.save(out_dir+'/npy/file_names.npy', np.array(file_names))

def make_data(x_dir, y_dir, out_dir, out_names, params):
    x_name, y_name = out_names 
    num_classes, weight, blur_y, norm_x, norm_y, aug, cat = params

    train_x = []
    train_y = []

    idx = 0
    file_names = np.load(out_dir+'/npy/file_names.npy')
    for file in file_names:
        x = np.array(kpi.load_img(x_dir+'/'+file+'.jpg'))
        y = np.load(y_dir+'/'+file+'.npy')
        x = x.astype(np.int16)
        y = y.astype(np.int16)

        if weight:
            y[y==0] = 600 #background
            y[y==1] = 600 #building-flooded
            y[y==2] = 600 #building-non-flooded 
            y[y==3] = 600 #road-flooded
            y[y==4] = 1   #road-non-floaded
            y[y==5] = 600 #water
            y[y==6] = 300 #tree
            y[y==7] = 600 #vehicle
            y[y==8] = 600 #pool
            y[y==9] = 50  #grass
        if blur_y:
            y = cv2.GaussianBlur(y, (5,5), 0.1)
            y = y.astype(np.int16)
        
        data = []
        data.append((x, y))
        if aug:
            data.append((img.flip(x, y)))
            data.append((img.zoom(x, y, 0.5)))
            data.append((img.shift(x, y, 0.2)))
            data.append((img.rotate(x, y, 5)))
            data.append((img.noise(x, y, .05)))

        for x, y in data:
            if norm_x:
                x = x.astype(np.float16)/255
            if norm_y:
                if weight:
                    y = y.astype(np.float16)/600
                else:
                    y = y.astype(np.float16)/(num_classes-1)

            # correct float values from transformations 
            if not norm_x and not norm_y:
                x = x.astype(np.int32)
                y = y.astype(np.int32)
            if norm_x:
                x[x>1] = 1
                x[x<0] = 0
            if norm_y:
                y[y>1] = 1
                y[y<0] = 0
            if not weight:
                y[y>(num_classes-1)] = 9
            if cat and not weight and not norm_y:
                y = (np.arange(num_classes) == y[..., None]).astype(int)
            train_x.append(x)
            train_y.append(y)

        if idx%100 == 0:
            print('file - ', idx)
        idx += 1

    print('Len: X - {}, Y - {}'.format(len(train_x), len(train_y)))
    print('Max: X - {}, Y - {}'.format(np.max(train_x), np.max(train_y)))
    print('Min: X - {}, Y - {}'.format(np.min(train_x), np.min(train_y)))

    print('\nSaving - ', x_name, y_name)
    np.save(out_dir+'/npy/'+x_name+'.npy', train_x)
    np.save(out_dir+'/npy/'+y_name+'.npy', train_y)
    print('Done')


#######################################################################################

x_dir = './train/X_Train_OG_256'
y_dir = './train/Y_Train_OG_256'
out_dir = './train'

num_classes = 10
weight = False
blur_y = True
norm_x = False
norm_y = False
aug = True
cat = True

params = (num_classes, weight, blur_y, norm_x, norm_y, aug, cat)
out_names = ('x_nwbnxnyac', 'y_nwbnxnyac')


#save_names(x_dir, 'jpg', out_dir)
make_data(x_dir, y_dir, out_dir, out_names, params)
