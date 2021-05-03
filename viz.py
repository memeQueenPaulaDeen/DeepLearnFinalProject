import copy
import math

import cv2 as cv
import os
import sys
import pandas as pd
import re
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import keras as k

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



class imageHelper():
    # the final purpose of this class will probably be to create image generators for training and visualisations

    def __init__(self, Weighting, GenerateStartStop=False,size2 = 512,crop2x = 2944, crop2y = 2944 ):
        self.pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.crop2x = crop2x
        self.crop2y = crop2y

        if crop2x != crop2y:
            w = RuntimeWarning('size 2 only enforced on X')
            warnings.warn(w)

        self.div = self.crop2x/size2

        floodPath = os.path.join(self.pwd,'Labeled','Flooded')
        NonfloodPath = os.path.join(self.pwd,'Labeled','Non-Flooded')

        #link all file locations in one convinent dataframe
        self.df = self.consumeDir(floodPath)
        self.df = self.df.append(self.consumeDir(NonfloodPath))


        # add our own labels # obsticle, grass, and road
        self.segLabelsDF = pd.read_csv(os.path.join(self.pwd,'classMap.csv'))

        self.segLabelsDF['Nav Class Name'] = self.segLabelsDF['Class Name']
        self.segLabelsDF.loc[~ self.segLabelsDF['Class Name'].isin(Weighting.keys()),'Nav Class Name'] = 'Obstacle'
        self.segLabelsDF['nav Weight'] =  self.segLabelsDF['Nav Class Name'].apply(lambda x: Weighting[x]) # add the weighting definied as a param to this class
        self.scaleing = self.segLabelsDF['nav Weight'].max()

        ### define the start and stop locations for each of the pics
        startStopLoc = os.path.join(self.pwd,'startStop.csv')
        if GenerateStartStop:
            self.runStartStopHelper(startStopLoc)

        assert os.path.exists(startStopLoc), 'Start and stop locations must be defined first'
        ssdf = pd.read_csv(startStopLoc)
        self.df['sx'] = ssdf['sx']
        self.df['sy'] = ssdf['sy']
        self.df['ex'] = ssdf['ex']
        self.df['ey'] = ssdf['ey']

        self.model = None

    def defineEndPoints(self,i,mask,div,img):

        xidx = 1
        yidx = 0

        FullColorMask = self.getColorForSegMap(mask)
        both = np.concatenate((cv.resize(i, (i.shape[xidx] // div, i.shape[yidx] // div)),
                               cv.resize(FullColorMask,
                                         (FullColorMask.shape[xidx] // div, FullColorMask.shape[yidx] // div))),
                              axis=1)

        name = 'original image left, segmented right for ' + img
        cv.imshow(name, both)

        startStopArr = [(-1, -1), (-1, 1)]

        def handleClickEvent(event, x, y, flags, params):
            # see https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/

            # checking for left mouse clicks
            if event == cv.EVENT_LBUTTONDOWN:
                # displaying the coordinates
                # on the image window
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(both, "start: " + str(x) + ',' +
                           str(y), (x, y), font,
                           1, (255, 0, 0), 2)
                startStopArr[0] = (x, y)
                cv.imshow(name, both)

            # checking for right mouse clicks
            if event == cv.EVENT_RBUTTONDOWN:
                # displaying the coordinates
                # on the Shell

                # displaying the coordinates
                # on the image window
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(both, "end: " + str(x) + ',' +
                           str(y), (x, y), font,
                           1, (0, 0, 255), 2)
                startStopArr[1] = (x, y)
                cv.imshow(name, both)

        cv.setMouseCallback(name, handleClickEvent)

        cv.waitKey(0)
        cv.destroyAllWindows()

        assert startStopArr != [(-1, -1), (-1, 1)], 'failed to update start Loc'
        row = {'img': img,
               'sx': startStopArr[0][0],
               'sy': startStopArr[0][1],
               'ex': startStopArr[1][0],
               'ey': startStopArr[1][1], }
        return row

    def runStartStopHelper(self,startStopLoc):

        LocationDf = pd.DataFrame([])
        imgs = self.df.img.values


        for img in imgs:

            i, mask = self.getImageMaskPair(img)
            div = 4

            row = self.defineEndPoints(i,mask,div,img)

            LocationDf = LocationDf.append(row,ignore_index=True)

            LocationDf.to_csv(startStopLoc)


        print("Start stop assigned for all images exiting")





    def getColorForSegMap(self,mask: np.array) -> np.array:
        # in BGR for open cv lib Human readable
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

        #Translate encodeing to human readable
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

        result = np.zeros((mask.shape[0],mask.shape[1],3))

        for m in m2Class.keys():
            result[mask == m] = cmapFullClass[m2Class[m]]

        return result.astype('uint8')



    def generate1dMask(self,maskLoc):

        dir = os.path.join(maskLoc,os.pardir,os.pardir,'1dmask')
        Path(dir).mkdir(parents=True, exist_ok=True)
        dir = os.path.abspath(dir)
        oneDMaskLoc = os.path.join(dir,maskLoc.split(os.sep)[-1])
        if not os.path.exists(oneDMaskLoc):# only need to do this once
            m3 = cv.imread(maskLoc)
            m1 = m3[:,:,0]
            cv.imwrite(oneDMaskLoc,m1)

        return oneDMaskLoc



    def consumeDir(self,dir: str) -> pd.DataFrame:

        result = pd.DataFrame([])
        imgDir = os.path.join(dir,'image')
        maskDir = os.path.join(dir,'mask')
        classType = dir.split(os.path.sep)[-1]


        for img in os.listdir(imgDir):

            num = re.findall('[0-9]+',img)
            assert len(num) == 1, 'Image name missformated: ' + img
            num = num[0]
            maskLoc = os.path.join(maskDir,num+"_lab.png")

            if not os.path.exists(maskLoc):
                w = RuntimeWarning('Could not find mask: ' + maskLoc + 'for the corresponding image: ' + img)
                warnings.warn(w)
                continue


            row = {'img':img,
                   'imgLoc':os.path.join(imgDir,img),
                   'maskLoc':maskLoc,
                   'classType': classType}
            result = result.append(row,ignore_index=True)

        return result

    def getStartAndStopLocForImag(self,img):
        simg = self.df.loc[self.df.img == img]
        sx = int(simg.sx.values[0])
        sy = int(simg.sy.values[0])
        ex = int(simg.ex.values[0])
        ey = int(simg.ey.values[0])

        return sx,sy,ex,ey

    def plotImgAndMask(self,img:str):
        i, mask = self.getImageMaskPair(img)
        sx, sy, ex, ey = self.getStartAndStopLocForImag(img)

        xidx = 1
        yidx = 0

        FullColorMask = self.getColorForSegMap(mask)
        both = np.concatenate((cv.resize(i,(i.shape[xidx],i.shape[yidx])),
                               cv.resize(FullColorMask,(FullColorMask.shape[xidx],FullColorMask.shape[yidx]))),
                              axis=1)

        name = 'original image left, segmented right for '+ img

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(both, "s", (sx, sy), font,
                   1, (255, 0, 0), 2)
        cv.putText(both, "s", (sx+i.shape[xidx], sy), font,
                   1, (255, 0, 0), 2)

        cv.putText(both, "E", (ex, ey), font,
                   1, (255, 0, 0), 2)
        cv.putText(both, "E", (ex + i.shape[xidx], ey), font,
                   1, (255, 0, 0), 2)

        cv.imshow(name, both)

        cv.waitKey(0)

    def getImageMaskPair(self, img:str, crop ='center') -> (np.array, np.array):
        irow = self.df.loc[self.df.img == img]
        img = cv.imread(irow.imgLoc.values[0])
        mask = cv.imread(irow['maskLoc'].values[0],cv.IMREAD_GRAYSCALE)
        # the images are not all the same size as the 4000x3000 mask crop image to match

        assert crop == 'center' or crop == 'left' or crop == 'right'
        if crop == 'left':
            img = img[0:self.crop2y, 0:self.crop2x]
            mask = mask[0:self.crop2y, 0:self.crop2x]
        if crop == 'center':
            cx = mask.shape[1]//2
            cy = mask.shape[0]//2

            img = img[cy-math.floor(self.crop2y / 2):cy + math.ceil(self.crop2y / 2),
                      cx-math.floor(self.crop2x / 2):cx + math.ceil(self.crop2x / 2)]
            mask = mask[cy - math.floor(self.crop2y / 2):cy + math.ceil(self.crop2y / 2),
                  cx - math.floor(self.crop2x / 2):cx + math.ceil(self.crop2x / 2)]
        if crop == 'right':
            img = img[mask.shape[0] - 1 - self.crop2y:mask.shape[0] - 1,
                  mask.shape[1] - 1 - self.crop2x:mask.shape[1] - 1]
            mask = mask[mask.shape[0] - 1 - self.crop2y:mask.shape[0] - 1,
                   mask.shape[1] - 1 - self.crop2x:mask.shape[1] - 1]

        assert img.shape[1] == self.crop2x and img.shape[0] == self.crop2y and \
               mask.shape[1] == self.crop2x and mask.shape[0] == self.crop2y

        # resize for reasonable plot
        return cv.resize(img, (int(self.crop2x // self.div), int(self.crop2y // self.div))), cv.resize(mask, (int(self.crop2x // self.div), int(self.crop2y // self.div)))

    def getValidBorderingIdx(self, col, row, mask, costMap):

        result = []
        colIdx, rowIdx = 1,0

        #check left idx good
        if col - 1 >= 0:
            nextCost = self.getCost(col-1,row,mask,costMap[row,col]) #todo: need to be able to accept any arbitrary weight array.

            #we found a better value so we update the wavefront and cost map
            if costMap[row,col-1] > nextCost:
                result.append((col - 1, row,nextCost))

        #check if right idx good
        if col+1 < costMap.shape[colIdx]:
            nextCost = self.getCost(col+1,row,mask,costMap[row,col])

            if costMap[row,col+1] > nextCost:
                result.append((col + 1, row,nextCost))

        #check up
        if row-1 >= 0:
            nextCost = self.getCost(col,row-1,mask,costMap[row,col])

            if costMap[row-1,col] > nextCost:
                result.append((col, row - 1,nextCost))
        #check down
        if row+1 < costMap.shape[rowIdx]:
            nextCost = self.getCost(col,row+1,mask,costMap[row,col])

            if costMap[row+1,col] > nextCost:
                result.append((col, row + 1,nextCost))

        return result


    def getCost(self,newCol,newRow,mask,currentCost):
        stepCost = mask[newRow,newCol]
        # detect normilized cost map
        # if scaled:
        #     stepCost = stepCost * self.scaleing

        return currentCost + stepCost

    def getWaveFrontCostForMask(self, img,x, y,plottingUpSample=2,calc_downSample = 2):

        xidx, yidx = 1, 0

        #Increase efficency by downsample then upsample
        i, mask = self.getImageMaskPair(img)
        row = self.defineEndPoints(cv.resize(i,(i.shape[xidx]*plottingUpSample,i.shape[yidx]*plottingUpSample)),
                                   cv.resize(mask,(mask.shape[xidx]*plottingUpSample,mask.shape[yidx]*plottingUpSample)),
                                   1,
                                   img)

        scaleFactor = plottingUpSample * calc_downSample
        sx, sy, ex, ey = row['sx'] // scaleFactor, row['sy'] // scaleFactor, row['ex'] // scaleFactor, row['ey'] // scaleFactor


        waveFront = set()
        initVal = 9999999999999
        costMap = np.ones((y.shape[yidx]//calc_downSample, y.shape[xidx]//calc_downSample)) * initVal

        costMap[ey,ex] = 0
        waveFront.add((ex,ey,0))

        count = 0
        while len(waveFront) > 0:
            col, row, currentCost = waveFront.pop()
            for trip in self.getValidBorderingIdx(col, row, cv.resize(y, (y.shape[xidx]//calc_downSample, y.shape[yidx]//calc_downSample)), costMap):#,interpolation=cv.INTER_NEAREST),costMap):
                waveFront.add(trip)
                # update cost map
                c, r, d = trip
                costMap[r,c] = d


            if count % 1000 == 0:


                hm = copy.deepcopy(costMap)
                hm[hm==initVal] = hm[hm!=initVal].max()
                hm = hm/hm.max()*255
                hm = cv.applyColorMap(hm.astype('uint8'), cv.COLORMAP_HOT)
                cv.imshow('heatMap for ' + img,cv.resize(hm,(hm.shape[xidx]*scaleFactor,hm.shape[yidx]*scaleFactor)))
                cv.waitKey(1)
                #cv.destroyAllWindows()
                print(count)

            count = count+1

        cv.waitKey(0)
        cv.destroyAllWindows()

        both = np.concatenate((cv.resize(x,(x.shape[xidx]//calc_downSample,x.shape[yidx]//calc_downSample)),hm),axis=1)
        pathColor = (247,5,239)#pink

        pos = (sx,sy)

        while pos != (ex,ey):
            xpos = pos[0]
            ypos = pos[1]
            both[ypos,xpos] = pathColor
            both[ypos,xpos+x.shape[xidx]//calc_downSample] = pathColor
            pos = self.gradDesc(xpos,ypos,costMap)

            name = 'original image left, distance to goal on right for ' + img

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(both, "s", (sx, sy), font,
                       1/calc_downSample, (55, 255, 55), 1)
            cv.putText(both, "s", (sx + x.shape[xidx]//calc_downSample, sy), font,
                       1/calc_downSample, (55, 255, 55), 1)

            cv.putText(both, "E", (ex, ey), font,
                       1/calc_downSample, (55, 255, 55), 1)
            cv.putText(both, "E", (ex + x.shape[xidx]//calc_downSample, ey), font,
                       1/calc_downSample, (55, 255, 55), 1)

            cv.imshow(name, cv.resize(both,(both.shape[xidx]*scaleFactor,both.shape[yidx]*scaleFactor)))

            cv.waitKey(1)

        self.predict(img,plot=True,destroy=False)

        cv.waitKey(0)
        cv.destroyAllWindows()

        return costMap

    def gradDesc(self,col,row,costMap):

        result = []
        colIdx, rowIdx = 1, 0

        # check left idx good
        if col - 1 >= 0:
            nextCost = costMap[row,col-1]
            # we found a better value so we update the wavefront and cost map
            if costMap[row, col] > nextCost:
                result.append((col - 1, row, nextCost))

        # check if right idx good
        if col + 1 < costMap.shape[colIdx]:
            nextCost = costMap[row, col+1]

            if costMap[row, col] > nextCost:
                result.append((col + 1, row, nextCost))

        # check up
        if row - 1 >= 0:
            nextCost = costMap[row-1, col]

            if costMap[row, col] > nextCost:
                result.append((col, row - 1, nextCost))
        # check down
        if row + 1 < costMap.shape[rowIdx]:
            nextCost =  costMap[row+1, col]

            if costMap[row, col] > nextCost:
                result.append((col, row + 1, nextCost))


        if len(result) == 0:
            w = RuntimeWarning('lower cost not found downsampling for wavefront may be to agressive')
            warnings.warn(w)
            result = (col+np.random.randint(0,2),row+np.random.randint(0,2))
        else:
            result = result[np.random.randint(0,len(result))]

        return result

    def getTrainEx(self,img,normalize = True,blurKsize = 1,crop = 'center') -> tuple:
        X, mask = self.getImageMaskPair(img,crop = crop)
        ClassToWeightMapping = self.segLabelsDF['nav Weight'].to_dict()
        Y_gt = np.zeros((mask.shape[0], mask.shape[1]))

        for m in ClassToWeightMapping.keys():
            Y_gt[mask == m] = ClassToWeightMapping[m]

        if normalize:
            Y_gt = Y_gt/self.scaleing

        Y_gt = cv.blur(Y_gt,(blurKsize,blurKsize))

        return X, Y_gt

    def saveTrainingExamples(self,regen=False):

        X_trainDir = os.path.join(self.pwd, 'X_Train_256/data')
        Y_trainDir = os.path.join(self.pwd,'Y_Train_256')
        Y_trainNormDir = os.path.join(self.pwd,'Y_TrainNorm_256')
        Y_trainBlurDir = os.path.join(self.pwd,'Y_TrainBlur_256')
        Y_trainNormBlurDir = os.path.join(self.pwd,'Y_TrainNormBlur_256')

        dirs = [X_trainDir,Y_trainDir,Y_trainNormDir,Y_trainBlurDir,Y_trainNormBlurDir]

        AllExist = True
        for dir in dirs:
            AllExist = AllExist and os.path.exists(dir)
            Path(dir).mkdir(parents=True, exist_ok=True)

        if not AllExist or regen:
            print('SAVE XY')
            self.df.img.apply(lambda img: self.saveXY(img,X_trainDir,Y_trainDir,normalize=False,blurKsize=1))

            print('SAVE X, Y normalized')
            self.df.img.apply(lambda img: self.saveXY(img,X_trainDir,Y_trainNormDir,normalize=True,blurKsize=1))

            print('SAVE X, Y blurred')
            self.df.img.apply(lambda img: self.saveXY(img,X_trainDir,Y_trainBlurDir,normalize=False,blurKsize=5))

            print('SAVE X, Y normalized and blurred')
            self.df.img.apply(lambda img: self.saveXY(img,X_trainDir,Y_trainNormBlurDir,normalize=True,blurKsize=5))



    def saveXY(self,img,X_trainDir,Y_trainDir,normalize = True,blurKsize = 1):
        imgPart = img.split('.')[0]
        ext = img.split('.')[1]

        #x, y = self.getImageMaskPair(img,crop='left')
        x, y = self.getTrainEx(img, normalize=normalize, blurKsize=blurKsize, crop='left')
        cv.imwrite(os.path.join(X_trainDir,imgPart+'l.'+ext),x)
        np.save(os.path.join(Y_trainDir,img.split('.')[0]+'l'),y)

        #x, y = self.getImageMaskPair(img)
        self.getTrainEx(img, normalize=normalize, blurKsize=blurKsize, crop='center')
        cv.imwrite(os.path.join(X_trainDir, imgPart + 'c.' + ext), x)
        np.save(os.path.join(Y_trainDir, img.split('.')[0] + 'c'), y)

        #x, y = self.getImageMaskPair(img,crop='right')
        x, y = self.getTrainEx(img, normalize=normalize, blurKsize=blurKsize, crop='right')
        cv.imwrite(os.path.join(X_trainDir, imgPart + 'r.' + ext), x)
        np.save(os.path.join(Y_trainDir, img.split('.')[0] + 'r'), y)

    def predictClass(self,img):
        x, y = self.getImageMaskPair(img)
        ypred = np.squeeze(self.model.predict(np.expand_dims(x, axis=0)))
        ypred = np.argmax(ypred,axis=2)

        yc = self.getColorForSegMap(y)
        ypredc = self.getColorForSegMap(ypred)


        cv.imshow('x', x)
        cv.imshow('y', yc)
        cv.imshow('ypred', ypredc)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def predict(self,img,plot = False,destroy = True):
        x, y = self.getTrainEx(img,blurKsize=11)
        ypred = np.squeeze(self.model.predict (np.expand_dims(x,axis=0)))

        if plot:
            hmy = y / y.max() * 255
            hmy = cv.applyColorMap(hmy.astype('uint8'), cv.COLORMAP_HOT)

            hmypred = ypred / ypred.max() * 255
            hmypred = cv.applyColorMap(hmypred.astype('uint8'), cv.COLORMAP_HOT)

            cv.imshow('x',x)
            cv.imshow('y',hmy)
            cv.imshow('ypred',hmypred)
            cv.waitKey(0)
            if destroy:
                cv.destroyAllWindows()

        return x, ypred






if __name__ == '__main__':

    #todo eventually will need to extend to show ground truth vs pred
    #todo implement generators for the input image (rescale) and output, predicted weight map.
    #todo either need to find a way to do transfer learning or use the semi supervised methods disccused in class


    Weighting = {
        'Obstacle': 600,
        'Tree': 300,
        'Grass': 50,
        'Road-non-flooded': 1
    }

    ih = imageHelper(Weighting,GenerateStartStop=False,size2=256)
    ih.saveTrainingExamples(regen=True)

    #print(ih.df)
    # x, y = ih.getTrainEx('8482.jpg', normalize=True,blurKsize=5)
    # ih.getWaveFrontCostForMask('8482.jpg', x, y)

    # x,y = ih.getTrainEx('6615.jpg',normalize=False,blurKsize=5)
    # ih.getWaveFrontCostForMask('6615.jpg',x,y,plottingUpSample=1)

    ih.model = k.models.load_model(os.path.join('models','m9'))
    print(ih.model.summary())

    # x, ypred = ih.predict('6750.jpg')
    # ih.getWaveFrontCostForMask('6750.jpg', x, ypred, plottingUpSample=2)


    for img in ih.df.img.values:
        ih.predict(img,plot=True)

    # for img in ih.df.img.values:
    #     x, ypred = ih.predict(img)
    #     ih.getWaveFrontCostForMask(img, x, ypred, plottingUpSample=2)

