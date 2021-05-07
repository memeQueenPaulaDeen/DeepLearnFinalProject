import numpy as np
import cv2 as cv


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

def getWaveFrontCostForMask(self, img, mask, x, y,plottingUpSample=2,calc_downSample = 2):

    xidx, yidx = 1, 0

    #Increase efficency by downsample then upsample
    #i, mask = self.getImageMaskPair(img)
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

