import abc
import copy
import itertools
import math
import os
import random
from abc import abstractmethod

import keras as k
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import cv2 as cv
import threading
from scipy import ndimage as nd


class DataSet(metaclass=abc.ABCMeta):


    classes = []
    class_mask_values = []
    navi_classes = []
    weights = []

    def __init__(self,folderPath,X_dir,Y_dir,img_shape,num_cat):

        if folderPath is not None:
            self.x_path = os.path.join(folderPath,X_dir)
            self.y_path = os.path.join(folderPath,Y_dir)
        else:
            print("No Folder Set for dataset")


        self.img_shape = img_shape
        self.num_cat = num_cat

        self.classes2mask = dict(zip(self.classes, self.class_mask_values))
        self.mask2classes = dict(zip(self.classes, self.class_mask_values))

        self.navi_class_weighting = dict(zip(self.navi_classes, self.weights))
        self.class_weighting = dict(zip(self.classes, self.weights))

        self.mask2class_encoding = dict(zip(self.class_mask_values,np.arange(len(self.class_mask_values))))
        self.class_encoding2Mask = dict(zip(np.arange(len(self.class_mask_values)),self.class_mask_values))

    # def __instancecheck__(cls, instance):
    #     return cls.__subclasscheck__(type(instance))

    @classmethod
    def __subclasshook__(cls, subclass):
        #maybe its better to do conditional logic and get different errors not 100% sure how this works
        return (hasattr(subclass, 'getPartion') and
                callable(subclass.getPartion) and
                hasattr(subclass, 'classes') and
                len(subclass.classes) > 0 and
                hasattr(subclass, 'class_mask_values') and
                len(subclass.class_mask_values) > 0 and
                hasattr(subclass, 'navi_classes') and
                len(subclass.navi_classes) > 0 and
                hasattr(subclass, 'weights') and
                len(subclass.weights) > 0
                or
                NotImplemented)


    @abstractmethod
    def getPartition(self,val:float,test:float) -> dict:
        """Load data set into test train and val dictionary"""
        raise not NotImplementedError

    def decodeOneHot2Mask(self, pred):
        img2decode = np.argmax(pred,axis=2)
        r = img2decode
        g = copy.deepcopy(img2decode)
        b = copy.deepcopy(img2decode)
        for classNum, mask_val in self.class_encoding2Mask.items():
            r[r==classNum] = mask_val[0]
            g[g==classNum] = mask_val[1]
            b[b==classNum] = mask_val[2]

        return np.dstack((r,g,b)).astype(np.uint8)

    def costMapFromEncoded(self, iy, blur_K_size=25, blur_sigma=7):
        ### Returns the normilized and blured cost map given a cat image mask

        # need to avoid concurrent mod
        iyc = copy.deepcopy(iy)
        # the encoding scheme should be ordered such that 0 -> first class in class array and weigh w0 corresponds to c0
        for enc_val in range(len(self.weights)):
            iy[iyc == enc_val] = self.weights[enc_val] / max(self.weights)

        iy = cv.GaussianBlur(iy, (blur_K_size, blur_K_size), blur_sigma)
        return iy

    def scaleNormedCostMap(self,cm):
        res = cm.astype(np.float32) * max(self.weights)
        return res

    def scaleNormedCostMapForPlot(self,cm):
        res = cm.astype(np.float32) *255
        return res.astype(np.uint8)

class SyntheticDataSet(DataSet):


    classes = ["Building","Road","Parking","nonFloodWater","FloodWater","map"]
    #class_mask_values = [(255, 0, 0),(45, 45, 45),(255, 90, 0),(0, 0, 255),(114, 93, 71),(255, 255, 0)]
    class_mask_values = [(255, 0, 0),(45, 45, 45),(255, 90, 0),(0, 0, 255),(111, 63, 12),(255, 255, 0)]

    navi_classes = ["Obstacle","Paved","Paved","Obstacle","Obstacle","BackGround"]
    weights = [400,1,1,400,400,20]



    def getPartition(self, val, test):
        valFreq = math.ceil(1 / val)
        testFreq = math.ceil(1 / test)

        # Check data quality and collect the names of all cites
        cities = []
        self.partion = {'train': [], 'test': [], 'val': []}
        for x in os.listdir(self.x_path):
            xfile = os.path.join(self.x_path, x)
            yfile = os.path.join(self.y_path, x)

            assert os.path.isfile(xfile) and os.path.isfile(yfile), \
                "missing file pair for " + str(xfile) + " and  " + str(yfile)

            # get the first image of every city and save the cities name
            if x.__contains__("_1.png"):
                cities.append(x[:-6])

        # split the cities into test train and validation
        testnvalIdx = np.linspace(0, len(cities)-1, valFreq + testFreq, dtype=int)
        testCities = [cities[c] for c in testnvalIdx[:testFreq]]
        valCities = [cities[c] for c in testnvalIdx[testFreq:]]
        trainCities = [x for x in cities if x not in valCities and x not in testCities]
        print("Train Cities: " + str(trainCities))
        print("Val Cities: " + str(valCities))
        print("Test Cities: " + str(testCities))

        for x in os.listdir(self.x_path):
            c = x.split("_")[0]
            if c in valCities:
                self.partion['val'].append(x)
            elif c in testCities:
                self.partion['test'].append(x)
            else:
                self.partion['train'].append(x)

        return self.partion


class TemplateGenerator(tf.keras.utils.Sequence):



    def __init__(self, dataSet : DataSet, partition : str, batchSize : int ,shuffle=True,val=.15, test=.15,aug=True):
        assert partition == 'val' or partition == 'test' or partition == 'train'

        p = dataSet.getPartition(val,test)
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.X = p[partition]
        self.Y = p[partition]
        self.img_shape = dataSet.img_shape
        self.x_path = dataSet.x_path
        self.y_path = dataSet.y_path
        self.dataSet= dataSet
        self.aug = aug

        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)





    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @abstractmethod
    def generateBatch(self, batch_X : list) -> tuple:
        raise not NotImplementedError

    def __len__(self):
        return len(self.X)//self.batchSize

    def __getitem__(self, index):

        batchIdxs = self.indexes[index*self.batchSize:(index+1)*self.batchSize]
        batch_x = [self.X[i] for i in batchIdxs]

        x , y = self.generateBatch(batch_x)

        return x,y

    def noise(self,x, y, cut):
        c = int(255 * cut)
        n1 = np.random.randint(0, 255, self.img_shape)
        n2 = np.random.randint(0, 255, self.img_shape)

        n1[n1 > c] = 0
        n2[n2 < 255 - c] = 0
        x = x + (n1 + n2)
        x[x > 255] = 255
        return x, y

    def rotate(self,x, y, angle):

        # if len(y.shape) == 2:
        #     y = np.expand_dims(y, axis=2)

        angle = int(random.uniform(-angle, angle))
        h, w = x.shape[:2]

        A = cv.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
        x = cv.warpAffine(x, A, (w, h))
        y = cv.warpAffine(y, A, (w, h))

        return x,y

    def zoom(self,x, y, range):
        # https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
        assert range < 1 and range > 0, 'Value should be less than 1 and greater than 0'
        range = random.uniform(range, 1)

        # if len(y.shape) == 2:
        #     y = np.expand_dims(y, axis=2)

        h, w = x.shape[:2]
        h_taken = int(range * h)
        w_taken = int(range * w)
        h_start = random.randint(0, h - h_taken)
        w_start = random.randint(0, w - w_taken)
        x = x[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
        y = y[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
        x = self.fill(x, h, w)
        y = self.fill(y, h, w)

        return x, y

    def fill(self,img, h, w):
        # https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
        img = cv.resize(img, (h, w), cv.INTER_CUBIC)
        return img

class CategoricalSyntheticGenerator(TemplateGenerator):

    def generateBatch(self,batch_X):
        x = np.empty((self.batchSize,*self.img_shape))
        y = np.empty((self.batchSize,self.img_shape[0],self.img_shape[1],self.dataSet.num_cat))

        # x_globe = [(0,0) for a in range(len(batch_X))]
        # y_globe = [(0,0) for a in range(len(batch_X))]

        def handleSingle(self,imgFile, idx):
            ix = k.preprocessing.image.load_img(os.path.join(self.x_path, imgFile))
            ix = k.preprocessing.image.img_to_array(ix)

            iy = k.preprocessing.image.load_img(os.path.join(self.y_path, imgFile))
            iy = k.preprocessing.image.img_to_array(iy)

            if self.aug:
                ix, iy = self.zoom(ix, iy, .3)
                ix,iy = self.noise(ix,iy,.035)
                ix,iy = self.rotate(ix,iy,10)

            # there is some noise in the unity data so need to infer bad labels
            # trying to fill by the closest

            # unfortunatly even after cleaning this is still needed as the augmentation process also can change the mask values on the boundary between classes
            # this has proven to be a fairly effective way to deal with this headache
            #

            foobarMask = None
            for mask_val in self.dataSet.mask2class_encoding:
                if foobarMask is None:
                    foobarMask = np.any(iy[:, :] != mask_val, axis=2)
                else:
                    foobarMask = np.logical_and(foobarMask, np.any(iy[:, :] != mask_val, axis=2))

            for mask_val, enc in self.dataSet.mask2class_encoding.items():
                iy[np.all(iy == mask_val, axis=2)] = enc

            # ideally we need to figure out how to make unity behave not shadding flat??
            indices = nd.distance_transform_edt(foobarMask, return_distances=False, return_indices=True)
            iy = iy[tuple(indices)]

            # only need one dimension
            iy = iy[:, :, 0]

            # there is some noise in the unity data so need to infer bad labels
            # Or there is noise added durring the augmentation process
            assert len(iy[iy > self.dataSet.num_cat]) == 0, "data quality get rekt"

            iy = tf.keras.utils.to_categorical(iy, self.dataSet.num_cat)
            #iy = (np.arange(self.dataSet.num_cat) == iy[...,None]-1).astype(int)

            x[idx] = ix
            y[idx] = iy



        threads = []
        idx = 0
        for imgFile in batch_X:

            t = threading.Thread(target=handleSingle,args=(self,imgFile,idx))
            t.start()
            idx = idx + 1
            threads.append(t)

        for t in threads:
            t.join()

        return x, y


class RegressionSyntheticGenerator(TemplateGenerator):

    def generateBatch(self,batch_X):
        x = np.empty((self.batchSize,*self.img_shape))
        y = np.empty((self.batchSize,self.img_shape[0],self.img_shape[1]))

        idx = 0
        for imgFile in batch_X:

            ix = k.preprocessing.image.load_img(os.path.join(self.x_path, imgFile))
            ix = k.preprocessing.image.img_to_array(ix)

            iy = k.preprocessing.image.load_img(os.path.join(self.y_path, imgFile))
            iy = k.preprocessing.image.img_to_array(iy)

            if self.aug:
                ix, iy = self.zoom(ix,iy,.3)
                ix, iy = self.noise(ix, iy, .035)
                ix, iy = self.rotate(ix, iy, 10)

            # there is some noise in the unity data so need to infer bad labels
            # trying to fill by the closest

            foobarMask = None
            for mask_val in self.dataSet.mask2class_encoding:
                if foobarMask is None:
                    foobarMask = np.any(iy[:, :] != mask_val, axis=2)
                else:
                    foobarMask = np.logical_and(foobarMask, np.any(iy[:, :] != mask_val, axis=2))

            for mask_val, enc in self.dataSet.mask2class_encoding.items():
                iy[np.all(iy == mask_val, axis=2)] = enc

            # ideally we need to figure out how to make unity behave not shadding flat??
            indices = nd.distance_transform_edt(foobarMask, return_distances=False, return_indices=True)
            iy = iy[tuple(indices)]

            # only need one dimension
            iy = iy[:, :, 0]

            # there is some noise in the unity data so need to infer bad labels
            # Or there is noise added durring the augmentation process
            assert len(iy[iy > self.dataSet.num_cat]) == 0, "data quality get rekt"

            iy = self.dataSet.costMapFromEncoded(iy)


            x[idx] = ix
            y[idx] = iy


            idx = idx + 1




        return x, y

