import cv2 as cv
import os
import numpy as np
import mosaic_support as ms
import UnityServer
import time

from SuperGluePretrainedNetwork.SuperGlueTest import SuperGlueMatcher


import sys
sys.path.append(os.path.join("C:\\","Users","samiw","OneDrive","Desktop","Desktop","VT","Research","imageStitch","SuperPoint","superpoint"))
from match_features_demo import SuperPointHomo



class SiftBasedStitcher:

    def __init__(self):

        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=1)
        self.search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(self.index_params, self.search_params)


    def getDesAndKp(self,img):
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        return kp, des

    def getMatchesAndH(self,nextDes,currDes, nextKp, currKp,isaffine):


        matches = self.flann.knnMatch(nextDes, currDes, k=2)
        matches = self.ratioTest(matches)

        H, mask = ms.compute_homography(nextKp, currKp, matches, affine=isaffine)

        ## adding remove outliers from saved matches
        matches = [matches[i] for i in range(len(matches)) if mask[i]]

        return H, matches

    def ratioTest(self,matches, r=.65):
        good = []
        for m, n in matches:
            if m.distance < r * n.distance:
                good.append(m)
        return good


class ImageStreamStitcher:

    def __init__(self,isaffine = True):


        # self.colorIDX = 0
        # self.greyIDX = 1
        # self.kpIDX = 2
        # self.desIDX = 3
        # self.mIDX = 4
        # self.hIDX = 5
        # self.hpIDX = 6
        # self.fIDX = 7
        # self.cornIDX = 8
        # self.pointCloudIdx = 9
        # self.kpUUIDIDX = 10

        self.isaffine = isaffine







    def consumeStream(self,s: UnityServer,costPredictionFunction,dsMax,plot =True,resizeH = 980,):
        # begin image and cost map stitching
        stitcher = SiftBasedStitcher()

        Hp_old = np.eye(3)
        p = None
        cm = None
        first2WorldFrame = None
        supPNT = None
        SGM = None

        curr = None
        while s.recivingStitch or len(s.imgs2Stitch) > 0:# cond still have things to stitch

            if curr is None:
                #get the first image #should be in bgr used cv.imdecode
                curr = s.imgs2Stitch.pop(0)


            if len(s.imgs2Stitch) == 0:
                #still expect to recive more images but need to wait for the next one

                time.sleep(.1)
                continue

            #should be in bgr used cv.imdecode
            next = s.imgs2Stitch.pop(0)

            if p is None:
                p = curr


            ############SIFT################
            # currKp, currDes = stitcher.getDesAndKp(curr)
            # nextKp, nextDes = stitcher.getDesAndKp(next)
            #
            # H, matches = stitcher.getMatchesAndH(nextDes, currDes, nextKp, currKp,self.isaffine)

            #############superPoints####################
            # if supPNT is None:
            #     supPNT = SuperPointHomo(None)
            #
            # H, matches, m_kp1, m_kp2 = supPNT.getResultsFromImgs(next, curr, self.isaffine)

            ####superGlue and points####
            if SGM is None:
                SGM = SuperGlueMatcher()

            H, mask = SGM.matchImgs(
                cv.cvtColor(curr, cv.COLOR_BGR2GRAY),
                cv.cvtColor(next, cv.COLOR_BGR2GRAY),
                self.isaffine)


            ######end H est

            Hp = np.matmul(Hp_old, H)

            if cm is None:
                cm = costPredictionFunction(cv.cvtColor(curr,cv.COLOR_BGR2RGB))

            cm_next = costPredictionFunction(cv.cvtColor(next,cv.COLOR_BGR2RGB))


            size, offset = ms.calculate_size(p.shape, next.shape, Hp, self.isaffine)

            p, _, _ = ms.merge_images(p, next, Hp, size, offset)
            cm, Hp, translation = ms.merge_costMaps(cm, cm_next, Hp, size, offset)


            Hp_old = Hp

            curr = next
            if plot:

                cv.imshow("pano", image_resize(p, height=resizeH))
                cv.imshow("Cost Map from raw", image_resize(cv.applyColorMap(ms.convert2Img(cm,dsMax), cv.COLORMAP_HOT), height=resizeH))

                cv.waitKey(1)

        cm[np.isnan(cm)] = cm[np.logical_not(np.isnan(cm))].max()

        cm[cm == 0] = cm[np.logical_not(np.isnan(cm))].max()



        return p, cm


def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
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
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def homoAug(img,rotMax=5,sheerMax=.03,projDistMax=.03):
    deg = np.random.random_sample()*2*rotMax -rotMax
    sheerx = np.random.random_sample()*2*sheerMax -sheerMax
    sheery = np.random.random_sample()*2*sheerMax -sheerMax
    proj1 = np.random.random_sample()*2*projDistMax - projDistMax
    proj2 = np.random.random_sample()*2*projDistMax - projDistMax

    rad = np.deg2rad(deg)
    Hr = np.array([
        [np.cos(rad) , -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0,0,1],
    ])
    Ha = np.array([
        [1,sheery,0],
        [sheerx,1,0],
        [0,0,1],
    ])
    Hp = np.array([
        [1,0,0],
        [0,1,0],
        [proj1,proj2,1],
    ])

    H_aug = Hr@Ha#@Hp

    # size, _ = ms.calculate_size((0,0),img.shape,H_aug,False)
    size = (480,480)
    return cv.warpPerspective(img,H_aug,size), H_aug