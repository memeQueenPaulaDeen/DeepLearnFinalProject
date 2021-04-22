import copy

import cv2 as cv
import os
import sys
import pandas as pd
import re
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class imageHelper():
    # the final purpose of this class will probably be to create image generators for training and visualisations

    def __init__(self, Weighting, GenerateStartStop=False,div = 8):
        self.pwd = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.div = div

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

    def getImageMaskPair(self,img:str,EnforcedX = 4000,EnforcedY = 3000) -> (np.array, np.array):
        irow = self.df.loc[self.df.img == img]
        img = cv.imread(irow.imgLoc.values[0])
        mask = cv.imread(irow['maskLoc'].values[0],cv.IMREAD_GRAYSCALE)
        # the images are not all the same size as the 4000x3000 mask crop image to match
        img = img[0:EnforcedY,0:EnforcedX]
        # resize for reasonable plot
        return cv.resize(img,(EnforcedX//self.div,EnforcedY//self.div)), cv.resize(mask,(EnforcedX//self.div,EnforcedY//self.div))

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

    def getWaveFrontCostForMask(self, img,x, y,plottingUpSample=2):

        xidx, yidx = 1, 0

        #Increase efficency by downsample then upsample
        i, mask = self.getImageMaskPair(img)
        row = self.defineEndPoints(cv.resize(i,(i.shape[xidx]*plottingUpSample,i.shape[yidx]*plottingUpSample)),
                                   cv.resize(mask,(mask.shape[xidx]*plottingUpSample,mask.shape[yidx]*plottingUpSample)),
                                   1,
                                   img)

        sx, sy, ex, ey = row['sx']//plottingUpSample, row['sy']//plottingUpSample, row['ex']//plottingUpSample, row['ey']//plottingUpSample


        waveFront = set()
        initVal = 9999999999999
        costMap = np.ones((y.shape[yidx], y.shape[xidx])) * initVal

        costMap[ey,ex] = 0
        waveFront.add((ex,ey,0))

        count = 0
        while len(waveFront) > 0:
            col, row, currentCost = waveFront.pop()
            for trip in self.getValidBorderingIdx(col, row, cv.resize(y, (y.shape[xidx], y.shape[yidx])), costMap):#,interpolation=cv.INTER_NEAREST),costMap):
                waveFront.add(trip)
                # update cost map
                c, r, d = trip
                costMap[r,c] = d


            if count % 1000 == 0:


                hm = copy.deepcopy(costMap)
                hm[hm==initVal] = hm[hm!=initVal].max()
                hm = hm/hm.max()*255
                hm = cv.applyColorMap(hm.astype('uint8'), cv.COLORMAP_HOT)
                cv.imshow('heatMap for ' + img,cv.resize(hm,(hm.shape[xidx]*plottingUpSample,hm.shape[yidx]*plottingUpSample)))
                cv.waitKey(1)
                #cv.destroyAllWindows()
                print(count)

            count = count+1

        cv.waitKey(0)
        cv.destroyAllWindows()

        both = np.concatenate((x,hm),axis=1)
        pathColor = (247,5,239)#pink

        pos = (sx,sy)

        while pos != (ex,ey):
            xpos = pos[0]
            ypos = pos[1]
            both[ypos,xpos] = pathColor
            both[ypos,xpos+x.shape[xidx]] = pathColor
            pos = self.gradDesc(xpos,ypos,costMap)

            name = 'original image left, distance to goal on right for ' + img

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(both, "s", (sx, sy), font,
                       1, (55, 255, 55), 2)
            cv.putText(both, "s", (sx + x.shape[xidx], sy), font,
                       1, (55, 255, 55), 2)

            cv.putText(both, "E", (ex, ey), font,
                       1, (55, 255, 55), 2)
            cv.putText(both, "E", (ex + x.shape[xidx], ey), font,
                       1, (55, 255, 55), 2)

            cv.imshow(name, cv.resize(both,(both.shape[xidx]*plottingUpSample,both.shape[yidx]*plottingUpSample)))

            cv.waitKey(1)

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

    def getTrainEx(self,img,normalize = True,blurKsize = 1) -> tuple:
        X, mask = self.getImageMaskPair(img)
        ClassToWeightMapping = self.segLabelsDF['nav Weight'].to_dict()
        Y_gt = np.zeros((mask.shape[0], mask.shape[1]))

        for m in ClassToWeightMapping.keys():
            Y_gt[mask == m] = ClassToWeightMapping[m]

        if normalize:
            Y_gt = Y_gt/self.scaleing

        Y_gt = cv.blur(Y_gt,(blurKsize,blurKsize))

        return X, Y_gt

    def saveTrainingExamples(self,regen=False):

        X_trainDir = os.path.join(self.pwd, 'X_Train')
        Y_trainDir = os.path.join(self.pwd,'Y_Train')
        Y_trainBlurDir = os.path.join(self.pwd,'Y_TrainBlur')

        dirs = [X_trainDir,Y_trainDir,Y_trainBlurDir]

        AllExist = True
        for dir in dirs:
            AllExist = AllExist and os.path.exists(dir)
            Path(dir).mkdir(parents=True, exist_ok=True)

        if not AllExist or regen:
            print('SAVE XY')
            self.df.img.apply(lambda img: self.saveXY(img,X_trainDir,Y_trainDir,normalize=True,blurKsize=1))

            print('SAVE X, Y blurred')
            self.df.img.apply(lambda img: self.saveXY(img,X_trainDir,Y_trainBlurDir,normalize=True,blurKsize=5))



    def saveXY(self,img,X_trainDir,Y_trainDir,normalize = True,blurKsize = 1):
        x,y = self.getTrainEx(img,normalize=normalize,blurKsize=blurKsize)
        cv.imwrite(os.path.join(X_trainDir,img),x)
        cv.imwrite(os.path.join(Y_trainDir,img),y)



if __name__ == '__main__':

    #todo eventually will need to extend to show ground truth vs pred
    #todo implement generators for the input image (rescale) and output, predicted weight map.
    #todo either need to find a way to do transfer learning or use the semi supervised methods disccused in class


    Weighting = {
        'Obstacle': 5000,
        'Tree': 300,
        'Grass': 50,
        'Road-non-flooded': 1
    }

    ih = imageHelper(Weighting,GenerateStartStop=False,div=12)
    ih.saveTrainingExamples(regen=False)

    #print(ih.df)
    # x, y = ih.getTrainEx('8482.jpg', normalize=True,blurKsize=5)
    # ih.getWaveFrontCostForMask('8482.jpg', x, y)

    x,y = ih.getTrainEx('6615.jpg',normalize=False,blurKsize=5)
    ih.getWaveFrontCostForMask('6615.jpg',x,y)