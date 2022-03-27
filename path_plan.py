
import copy
import math
import random
from Generators import DataSet, SyntheticDataSet

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


class WaveFront:


    xidx = 1
    yidx = 0
    initVal = 9999999999999

    def __init__(self,img : np.array ,pixelCostMap : np.array, dataSet : DataSet, calc_downSample : int ,ex : int = None,ey : int = None):

        self.img = img
        self.pixelCostMap = pixelCostMap
        # self.pixelCostMap = dataSet.scaleNormedCostMap(pixelCostMap)
        #self.pixelCostMapIMG = dataSet.scaleNormedCostMapForPlot(pixelCostMap)
        self.dataSet = dataSet
        self.waveCostMap = None
        self.ex = ex
        self.ey = ey
        self.calc_downSample = calc_downSample



    def stepDownGrad(self,col, row,waveCostMap):

        result = []
        colIdx, rowIdx = 1, 0

        # check left idx good
        if col - 1 >= 0:
            nextCost = waveCostMap[row, col - 1]
            # we found a better value so we update the wavefront and cost map
            if waveCostMap[row, col] > nextCost:
                result.append((col - 1, row, nextCost))

        # check if right idx good
        if col + 1 < waveCostMap.shape[colIdx]:
            nextCost = waveCostMap[row, col + 1]

            if waveCostMap[row, col] > nextCost:
                result.append((col + 1, row, nextCost))

        # check up
        if row - 1 >= 0:
            nextCost = waveCostMap[row - 1, col]

            if waveCostMap[row, col] > nextCost:
                result.append((col, row - 1, nextCost))
        # check down
        if row + 1 < waveCostMap.shape[rowIdx]:
            nextCost = waveCostMap[row + 1, col]

            if waveCostMap[row, col] > nextCost:
                result.append((col, row + 1, nextCost))

        if len(result) == 0:
            w = RuntimeWarning('lower cost not found downsampling for wavefront may be to agressive')
            warnings.warn(w)
            result = (col + np.random.randint(0, 2), row + np.random.randint(0, 2))
        else:
            result = result[np.random.randint(0, len(result))]

        return result


    def readWaveCostMapFromFile(self,loc):
        with open(loc,'rb') as f:
            self.waveCostMap = np.load(f)

        ey, ex = np.where(self.waveCostMap == self.waveCostMap.min())
        self.ex = ex[0]
        self.ey = ey[0]

    def writeWaveCostMapToFile(self,loc):
        with open(loc,'wb') as f:
            np.save(f,self.waveCostMap)

    def getHeatPlotForWave(self,plottingUpSample):
        assert self.waveCostMap is not None, "need to init wave front map first"
        scaleFactor = self.getScaleFactor(plottingUpSample)
        hm = copy.deepcopy(self.waveCostMap)
        hm[hm == WaveFront.initVal] = hm[hm != WaveFront.initVal].max()
        hm = hm / hm.max() * 255
        hm = cv.applyColorMap(hm.astype('uint8'), cv.COLORMAP_HOT)
        return cv.resize(hm, (int(hm.shape[WaveFront.xidx] * scaleFactor), int(hm.shape[WaveFront.yidx] * scaleFactor)))

    def getImgAtHeatPlotSize(self,plottingUpSample):
        scaleFactor = self.getScaleFactor(plottingUpSample)
        hrxC = self.img.copy()
        hrxC = cv.resize(hrxC, (int(hrxC.shape[WaveFront.xidx] // self.calc_downSample * scaleFactor), int(hrxC.shape[WaveFront.yidx] // self.calc_downSample * scaleFactor)))
        return hrxC

    def getPCMHeatMapPlot(self,plottingUpSample):
        scaleFactor = self.getScaleFactor(plottingUpSample)
        hrxC = self.pixelCostMap *255 / max(self.dataSet.weights)
        hrxC = cv.applyColorMap(hrxC.astype('uint8'), cv.COLORMAP_HOT)
        hrxC = cv.resize(hrxC, (int(hrxC.shape[WaveFront.xidx] // self.calc_downSample * scaleFactor), int(hrxC.shape[WaveFront.yidx] // self.calc_downSample * scaleFactor)))

        return hrxC

    def generateWaveCostMap(self,plotName : str = None, plot : bool = False,plottingUpSample =None):

        assert self.ex is not None and self.ey is not None, "Goal location is not set"

        waveFront = set()
        self.waveCostMap = np.ones((self.pixelCostMap.shape[WaveFront.yidx] // self.calc_downSample, self.pixelCostMap.shape[WaveFront.xidx] // self.calc_downSample)) * WaveFront.initVal

        self.waveCostMap[self.ey, self.ex] = 0
        waveFront.add((self.ex, self.ey, 0))

        mask = cv.resize(self.pixelCostMap, (self.pixelCostMap.shape[WaveFront.xidx] // self.calc_downSample, self.pixelCostMap.shape[WaveFront.yidx] // self.calc_downSample))
        assert np.all(mask > 0), "check data quality 0 should not be present in cost map"
        # enforce non 0
        # mask = mask + 1

        count = 0
        while len(waveFront) > 0:
            col, row, currentCost = waveFront.pop()
            for trip in self.getValidBorderingIdx(col, row,mask ):  # ,interpolation=cv.INTER_NEAREST),self.waveCostMap):
                waveFront.add(trip)
                # update cost map
                c, r, d = trip
                self.waveCostMap[r, c] = d

            if count % 1000 == 0:

                if plot:
                    assert plottingUpSample is not None and plotName is not None
                    cv.imshow('heatMap for ' + plotName, self.getHeatPlotForWave(plottingUpSample))
                    cv.waitKey(1)
                    # cv.destroyAllWindows()
                print(count)

            count = count + 1


    def getValidBorderingIdx(self,col, row, mask):

        result = []
        colIdx, rowIdx = 1, 0

        # check left idx good
        if col - 1 >= 0:
            nextCost = self.getCost(col - 1, row, mask, self.waveCostMap[row, col])

            # we found a better value so we update the wavefront and cost map
            if self.waveCostMap[row, col - 1] > nextCost:
                result.append((col - 1, row, nextCost))

        # check if right idx good
        if col + 1 < self.waveCostMap.shape[colIdx]:
            nextCost = self.getCost(col + 1, row, mask, self.waveCostMap[row, col])

            if self.waveCostMap[row, col + 1] > nextCost:
                result.append((col + 1, row, nextCost))

        # check up
        if row - 1 >= 0:
            nextCost = self.getCost(col, row - 1, mask, self.waveCostMap[row, col])

            if self.waveCostMap[row - 1, col] > nextCost:
                result.append((col, row - 1, nextCost))
        # check down
        if row + 1 < self.waveCostMap.shape[rowIdx]:
            nextCost = self.getCost(col, row + 1, mask, self.waveCostMap[row, col])

            if self.waveCostMap[row + 1, col] > nextCost:
                result.append((col, row + 1, nextCost))

        return result

    def getCost(self,newCol, newRow, mask, currentCost):
        stepCost = mask[newRow, newCol]
        # detect normilized cost map
        # if scaled:
        #     stepCost = stepCost * self.scaleing

        return currentCost + stepCost

    def userDefinedPoint(self, plottingUpSample, imgPlot =None, heatMapPlot=None, plotName=""):

        assert not (imgPlot is None and heatMapPlot is None), "An image must be provided"

        # FullColorMask = self.getColorForSegMap(mask)
        plotBoth = imgPlot is not None and heatMapPlot is not None

        if plotBoth:
            both = np.concatenate((imgPlot,heatMapPlot),axis=1)
            toPlot = both

            name = 'original image left, segmented right ' + plotName
            cv.imshow(name, both)
        else:
            if imgPlot is not None:
                name = "RGB image " + plotName
                cv.imshow(name,imgPlot)
                toPlot = imgPlot
            else:
                name = "HeatMap " + plotName
                cv.imshow(name,heatMapPlot)
                toPlot = heatMapPlot


        Arr = [(-1, -1)]

        def handleClickEvent(event, x, y, flags, params):
            # see https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/

            # checking for left mouse clicks
            if event == cv.EVENT_LBUTTONDOWN:
                # displaying the coordinates
                # on the image window
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(toPlot, str(x) + ',' +
                           str(y), (x, y), font,
                           1, (255, 0, 0), 2)
                Arr[0] = (x, y)
                cv.imshow(name, toPlot)



        cv.setMouseCallback(name, handleClickEvent)

        cv.waitKey(0)
        cv.destroyAllWindows()

        assert Arr != [(-1, -1)], 'failed to update start Loc'
        row = {
            'x': Arr[0][0],
            'y': Arr[0][1],
             }

        scaleFactor = self.getScaleFactor(plottingUpSample)
        return int(row['x']//scaleFactor), int(row['y']//scaleFactor)


    def manualSetEndPoints(self,plottingUpSample,imgPlot=None, heatMapPlot=None,plotName=""):
        self.ex, self.ey = self.userDefinedPoint(plottingUpSample,imgPlot, heatMapPlot,plotName=plotName)

    def fullWaveFrontNavEX(self,plottingUpSample,plotName = "test"):

        hm = self.getPCMHeatMapPlot(plottingUpSample)
        self.manualSetEndPoints(plottingUpSample,imgPlot=self.getImgAtHeatPlotSize(plottingUpSample),heatMapPlot=hm,plotName=plotName)
        self.generateWaveCostMap(plottingUpSample=plottingUpSample,plotName="WaveFrontHeatMap",plot=True)

        # self.writeWaveCostMapToFile('norfolkTestCostMap.npy')
        # self.readWaveCostMapFromFile('norfolkTestCostMap.npy')

        sx, sy = self.userDefinedPoint(plottingUpSample,imgPlot=self.getImgAtHeatPlotSize(plottingUpSample),heatMapPlot=self.getHeatPlotForWave(plottingUpSample),plotName=plotName)

        self.plotGlobalPath("Wave Front Global Path", plottingUpSample, sx, sy, imgPlot=self.getImgAtHeatPlotSize(plottingUpSample))


    def getScaleFactor(self,plottingUpSample):
        return plottingUpSample * self.calc_downSample

    def plotGlobalPath(self, plotName, plottingUpSample, sx, sy,
                       imgPlot =None, heatMapPlot=None, weightMap=None, plotImeadiate=False, pathColor = (247, 5, 239), cvWaitKey=0):

        assert not (imgPlot is None and heatMapPlot is None), "An image must be provided"

        # FullColorMask = self.getColorForSegMap(mask)
        plotBoth = imgPlot is not None and heatMapPlot is not None

        if plotBoth:
            both = np.concatenate((imgPlot, heatMapPlot), axis=1)
            toPlot = both
        else:
            if imgPlot is not None:
                toPlot = imgPlot
            else:
                toPlot = heatMapPlot


        pos = (sx, sy)

        if weightMap is not None:
            result = 0

        scaleFactor = self.getScaleFactor(plottingUpSample)
        while pos != (self.ex, self.ey):
            xpos = pos[0]
            ypos = pos[1]
            # both[ypos, xpos] = pathColor



            toPlot[int(ypos * scaleFactor), int(xpos * scaleFactor)] = pathColor
            if plotBoth:
                toPlot[ypos, xpos + self.img.shape[WaveFront.xidx] // self.calc_downSample] = pathColor

            pos = self.stepDownGrad(xpos, ypos,self.waveCostMap)

            if scaleFactor > 1:
                cv.line(toPlot, (int(xpos * scaleFactor), int(ypos * scaleFactor)),
                        (int(pos[0] * scaleFactor), int(pos[1] * scaleFactor)), pathColor, 1)
                if plotBoth:
                    cv.line(toPlot, (int(xpos * scaleFactor) + self.img.shape[WaveFront.xidx], int(ypos * scaleFactor)),
                            (int(pos[0] * scaleFactor) + self.img.shape[WaveFront.xidx], int(pos[1] * scaleFactor)), pathColor, 1)


            if weightMap is not None:
                # would this depend on the size? maybe should be weighted by downsaple or something?
                result = result + weightMap[ypos * self.calc_downSample, xpos * self.calc_downSample]



            font = cv.FONT_HERSHEY_SIMPLEX

            cv.putText(toPlot, "s", (sx, sy), font,
                       1 / self.calc_downSample, (55, 255, 55), 1)
            cv.putText(toPlot, "E", (self.ex, self.ey), font,
                       1 / self.calc_downSample, (55, 255, 55), 1)

            if plotBoth:
                cv.putText(toPlot, "s", (sx + self.img.shape[WaveFront.xidx] // self.calc_downSample, sy), font,
                           1 / self.calc_downSample, (55, 255, 55), 1)
                cv.putText(toPlot, "E", (self.ex + self.img.shape[WaveFront.xidx] // self.calc_downSample, WaveFront.ey), font,
                           1 / self.calc_downSample, (55, 255, 55), 1)



            if not plotImeadiate:
                cv.imshow(plotName, toPlot)

                cv.waitKey(1)

        cv.imshow(plotName, toPlot)
        cv.waitKey(cvWaitKey)
        if weightMap is not None:
            return toPlot, result*self.calc_downSample #I think the mul is good?
        return toPlot


    #############################might need to move some of below into image proc class

    def getLocalCostMapFromTemplate(self, templateImage, plottingUpSample,plotMatchedSubcrop = True):


        #Could add conditional logic for method of finding bound extent IE sift mathcing, simple match etc
        startX, startY, endX, endY = self.localizeUAVViewSimpleTemplateMatching(self.img, templateImage,plottingUpSample, plot=True)


        subImg = self.img[startY:endY, startX:endX]
        localCostMap = self.waveCostMap[startY // self.calc_downSample:endY // self.calc_downSample,
                       startX // self.calc_downSample:endX // self.calc_downSample]
        if plotMatchedSubcrop:
            cv.imshow("mathced sub", subImg)
            cv.waitKey(1)

        return localCostMap

    def localizeUAVViewSimpleTemplateMatching(self, pano, img, plottingUpSample, plot=False,showGlobalPath = True):
        pBnW = cv.cvtColor(pano, cv.COLOR_BGR2GRAY)
        imgBnW = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        tmatched = cv.matchTemplate(pBnW, imgBnW, cv.TM_CCOEFF_NORMED)

        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(tmatched)

        (startX, startY) = maxLoc
        endX = startX + imgBnW.shape[1]
        endY = startY + imgBnW.shape[0]

        if plot:
            p_bb = pano.copy()
            cv.rectangle(p_bb, (startX, startY), (endX, endY), (255, 0, 0), 3)
            scaleFactor = self.getScaleFactor(plottingUpSample)
            if showGlobalPath:
                self.plotGlobalPath("BB and global path plotted on Pano",
                                    plottingUpSample,
                                    (startX + imgBnW.shape[1]//2)//self.calc_downSample,
                                    (startY + imgBnW.shape[0]//2)//self.calc_downSample,
                                    imgPlot=cv.resize(p_bb, (int(p_bb.shape[WaveFront.xidx] // self.calc_downSample * scaleFactor), int(p_bb.shape[WaveFront.yidx] // self.calc_downSample * scaleFactor))),
                                    plotImeadiate=True,
                                    pathColor=(0,255,0),
                                    cvWaitKey=1
                                    )
            else:
                cv.imshow("pano", self.image_resize(p_bb, height=p_bb.shape[WaveFront.yidx//scaleFactor]))
            cv.imshow("template", img)
            cv.waitKey(1)

        
        return startX, startY, endX, endY

    def image_resize(self,image, width=None, height=None, inter=cv.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def getLocalPathFromLocalCostMap(self, sx, sy, localImg, localWaveFrontMat, weightMap=None, plot=True, pathColor = (247, 5, 239)):

        globalGoalAchived = False
        path = []
        def checkBoundsStopCondition(xpos, ypos, localWaveFrontMat, xidx, yidx):
            xshape = localWaveFrontMat.shape[xidx]
            yshape = localWaveFrontMat.shape[yidx]

            result = not (xpos < 1 or xpos > (xshape - 2) or ypos < 1 or ypos > (yshape - 2))
            return result

        pos = (sx, sy)

        xpos = int(pos[0])
        ypos = int(pos[1])

        if weightMap is not None:
            result = 0


        hrxC = localImg.copy()
        # hrxC = cv.resize(hrxC, (
        # int(hrxC.shape[xidx] // calc_downSample * scaleFactor), int(hrxC.shape[yidx] // calc_downSample * scaleFactor)))

        while localWaveFrontMat[ypos, xpos] != 0 and checkBoundsStopCondition(xpos, ypos, localWaveFrontMat, WaveFront.xidx,
                                                                              WaveFront.yidx):
            xpos = int(pos[0])
            ypos = int(pos[1])
            # both[ypos, xpos] = pathColor

            hrxC[int(ypos * self.calc_downSample), int(xpos * self.calc_downSample)] = pathColor

            pos = self.stepDownGrad(xpos, ypos, localWaveFrontMat)
            path.append([pos[0],pos[1]])

            if self.calc_downSample > 1:
                cv.line(hrxC, (int(xpos * self.calc_downSample), int(ypos * self.calc_downSample)),
                        (int(pos[0] * self.calc_downSample), int(pos[1] * self.calc_downSample)), pathColor, 1)

            if weightMap is not None:
                # would this depend on the size? maybe should be weighted by downsaple or something?
                result = result + weightMap[ypos * self.calc_downSample, xpos * self.calc_downSample]

            # cv.imshow("SubImagePath", hrxC)
            # cv.waitKey(1)

            if localWaveFrontMat[ypos, xpos] == 0:
                if (sx,sy) == pos:
                    globalGoalAchived = True

        if plot:
            cv.imshow("SubImagePath", hrxC)
            cv.waitKey(1)
        if weightMap is not None:
            return hrxC, result
        return hrxC, path, globalGoalAchived  # also need to return json of image cordinate watpoints 2 follow

if __name__ == '__main__':
    fdir = os.path.join("C:\\", "Users", "samiw", "OneDrive", "Desktop", "Desktop", "VT", "Research", "imageStitch",
                        "testOutPuts", "full_res")

    xpath = os.path.join(fdir, "X_pano.png")
    pixelCostpath = os.path.join(fdir, "cm_pano.npy")
    calc_downSample = 8
    plottingUpSample = 1 / 2

    img_shape = (480, 480, 3)
    num_cat = 6
    d = SyntheticDataSet(None, None, None, img_shape,num_cat)

    with open(pixelCostpath, 'rb') as f:
        pcm = np.load(f)

    ####DO NOT FEED TO KERAS LIKE THIS
    x = cv.imread(xpath)


    w = WaveFront(x,pcm,d,calc_downSample)
    w.fullWaveFrontNavEX(plottingUpSample)


    ###############save a wave cost map to file################

    # hm = w.getPCMHeatMapPlot(plottingUpSample)
    # w.manualSetEndPoints(plottingUpSample,imgPlot=w.getImgAtHeatPlotSize(plottingUpSample),heatMapPlot=hm,plotName="wave")
    # w.generateWaveCostMap(plottingUpSample=plottingUpSample,plotName="WaveFrontHeatMap",plot=True)
    # w.writeWaveCostMapToFile('norfolkTestCostMap.npy')
    #
    # w.readWaveCostMapFromFile('norfolkTestCostMap.npy')