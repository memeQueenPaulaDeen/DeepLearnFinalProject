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

    def __init__(self, Weighting, GenerateStartStop=False):
        self.pwd = os.path.dirname(os.path.abspath(sys.argv[0]))

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


    def runStartStopHelper(self,startStopLoc):

        LocationDf = pd.DataFrame([])
        imgs = self.df.img.values


        for img in imgs:

            i, mask = self.getImageMaskPair(img)
            div = 4
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
                    startStopArr[0] = (x,y)
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
            row = {'img':img,
                    'sx':startStopArr[0][0],
                   'sy':startStopArr[0][1],
                   'ex':startStopArr[1][0],
                   'ey':startStopArr[1][1],}
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

        div = 4
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

    def getImageMaskPair(self,img:str,div = 4,EnforcedX = 4000,EnforcedY = 3000) -> (np.array, np.array):
        irow = self.df.loc[self.df.img == img]
        img = cv.imread(irow.imgLoc.values[0])
        mask = cv.imread(irow['maskLoc'].values[0],cv.IMREAD_GRAYSCALE)
        # the images are not all exactly the same scale them to a fixed size before returning
        return cv.resize(img,(EnforcedX//div,EnforcedY//div)), cv.resize(mask,(EnforcedX//div,EnforcedY//div))

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
        stepCost = self.segLabelsDF['nav Weight'][mask[newRow,newCol]]
        return currentCost + stepCost

    def getWaveFrontCostForMask(self,img):

        xidx, yidx = 1, 0

        #Increase efficency by downsample then upsample

        i, mask = self.getImageMaskPair(img)
        sx, sy, ex, ey = self.getStartAndStopLocForImag(img)

        downSample = 4

        waveFront = set()
        initVal = 9999999999999
        costMap = np.ones((mask.shape[yidx]//downSample,mask.shape[xidx]//downSample)) * initVal

        costMap[ey//downSample,ex//downSample] = 0
        waveFront.add((ex//downSample,ey//downSample,0))

        count = 0
        while len(waveFront) > 0:
            col, row, currentCost = waveFront.pop()
            for trip in self.getValidBorderingIdx(col, row, cv.resize(mask,(mask.shape[xidx]//downSample,mask.shape[yidx]//downSample),interpolation=cv.INTER_NEAREST),costMap):
                waveFront.add(trip)
                # update cost map
                c, r, d = trip
                costMap[r,c] = d


            if count % 1000 == 0:


                hm = copy.deepcopy(costMap)
                hm[hm==initVal] = hm[hm!=initVal].max()
                hm = hm/hm.max()*255
                hm = cv.applyColorMap(hm.astype('uint8'), cv.COLORMAP_HOT)
                hm = cv.resize(hm,(mask.shape[xidx],mask.shape[yidx]),interpolation=cv.INTER_NEAREST)
                cv.imshow('heatMap for ' + img,hm)
                cv.waitKey(1)
                #cv.destroyAllWindows()
                print(count)

            count = count+1

        cv.waitKey(0)
        costMap = cv.resize(costMap,(mask.shape[xidx],mask.shape[yidx]),interpolation=cv.INTER_NEAREST)

        return costMap




if __name__ == '__main__':

    #todo decide on final image size assuimging training image is reduced to w//4, h//4
    #todo Impelment modified wave front
    #todo create heat map function show the travel cost given a start and end point
    #todo find a good set of weights for the classes probably dont want to count tree conver as an obsticle anymore
    #todo implement generators for the input image (rescale) and output, predicted weight map.
    #todo either need to find a way to do transfer learning or use the semi supervised methods disccused in class


    Weighting = {
        'Obstacle': 5000,
        'Tree': 300,
        'Grass': 50,
        'Road-non-flooded': 1
    }

    ih = imageHelper(Weighting,GenerateStartStop=False)
    #print(ih.df)
    # ih.plotImgAndMask('6706.jpg')
    # ih.getWaveFrontCostForMask('6706.jpg')

    ih.plotImgAndMask('7049.jpg')
    ih.getWaveFrontCostForMask('7049.jpg')