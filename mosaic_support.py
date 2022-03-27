#####################################################################

# Example : real-time mosaicking - supporting functionality

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Acknowledgements: bmhr46@durham.ac.uk (2016/17);
# Marc Pare, code taken from:
# https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py

# no claims are made that these functions are completely bug free

# Copyright (c) 2017-21 Toby Breckon, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################
import copy

import cv2
import numpy as np
from scipy import ndimage as nd

#####################################################################

# check if the OpenCV we are using has the extra modules available


def extra_opencv_modules_present():

    # we only need to check this once and remember the result
    # so we can do this via a stored function attribute (static variable)
    # which is preserved across calls

    if not hasattr(extra_opencv_modules_present, "already_checked"):
        (is_built, not_built) = cv2.getBuildInformation().split("Disabled:")
        extra_opencv_modules_present.already_checked = (
            'xfeatures2d' in is_built
            )

    return extra_opencv_modules_present.already_checked


def non_free_algorithms_present():

    # we only need to check this once and remember the result
    # so we can do this via a stored function attribute (static variable)
    # which is preserved across calls

    if not hasattr(non_free_algorithms_present, "already_checked"):
        (before, after) = cv2.getBuildInformation().split(
            "Non-free algorithms:")
        output_list = after.split("\n")
        non_free_algorithms_present.already_checked = ('YES' in output_list[0])

    return non_free_algorithms_present.already_checked

#####################################################################

# Takes an image and a threshold value and
# returns the SIFT/SURF features points (kp) and descriptors (des) of image
# (for SURF features - Hessian threshold of typically 400-1000 can be used)

# if SIFT/SURF does not work on your system, auto-fallback to ORB
# [this could be optimized for a specific system configuration,
# and also so as not to create these detector objects _every_ time ]


def get_features(img, thres):

    (major, minor, _) = cv2.__version__.split(".")
    if ((int(major) >= 4) and (int(minor) >= 4)):

        # if we have SIFT available then use it (in main branch of OpenCV)

        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

    elif (non_free_algorithms_present()):

        # if we have SURF available then use it (with Hessian Threshold =
        # thres)
        surf = cv2.xfeatures2d.SURF_create(thres)
        kp, des = surf.detectAndCompute(img, None)
        # check which features we have available

    else:

        # otherwise fall back to ORB (with Max Features = thres)
        orb = cv2.ORB_create(thres)
        kp, des = orb.detectAndCompute(img, None)

    return kp, des

#####################################################################

# Performs FLANN-based feature matching of descriptor from 2 images
# returns 'good matches' based on their distance
# typically number_of_checks = 50, match_ratio = 0.7

# if SURF does not work on your system, auto-fallback to ORB
# [this could be optimized for a specific system configuration]


def match_features(des1, des2, number_of_checks, match_ratio):

    # check which features we have available / are using

    (major, minor, _) = cv2.__version__.split(".")
    if (((int(major) >= 4) and (int(minor) >= 4)) or
            (non_free_algorithms_present())):

        # assume we are using SIFT/SURF points use
        index_params = dict(algorithm=1, trees=1)  # FLANN_INDEX_KDTREE = 1

    else:

        # if using ORB points (taken from:)
        # https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
        # N.B. "commented values are recommended as per the docs,
        # but it didn't provide required results in some cases"

        flann_index_lsh = 6
        index_params = dict(algorithm=flann_index_lsh,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2

    # set up and use a FLANN matcher (reset each time it is used)
    search_params = dict(checks=number_of_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []

    # as the available number of matches recovered varies with the scene
    # and hence the number features detected the following can fail under
    # certain conditions (i.e. not enough matches found).
    # suggestion 1: heavily filter / control number of feature + matches going
    # into this next section of code
    # suggestion 2: wrap the following in a try/catch construct
    # https://docs.python.org/3/tutorial/errors.html

    for (m, n) in matches:
        if m.distance < match_ratio * n.distance:  # filter out 'bad' matches
            good_matches.append(m)
    return good_matches

#####################################################################

# Computes and returns the homography matrix H relating the two sets
# of keypoints relating to image 1 (kp1) and (kp2)


def compute_homography(kp1, kp2, good_matches,affine):

    # set up point lists
    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


    # X[:, 0], X[:, 1] = X[:, 1], X[:, 0].copy()
    # Y[:, 0], Y[:, 1] = Y[:, 1], Y[:, 0].copy()
    # Z[:, 0], Z[:, 1] = Z[:, 1], Z[:, 0].copy()
    # compute the transformation using RANSAC to find homography
    if affine:
        homography, mask = cv2.estimateAffinePartial2D(pts1, pts2, None,cv2.RANSAC, 5)
        homography = np.r_[homography,np.array([0,0,1]).reshape(1,3)]
    else:
        homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5)
    return homography, mask

#####################################################################

# Calculates the required size for the mosaic based on the dimensions of
# two input images (provided as img.shape) and also homography matrix H
# returns new size and 2D translation offset vector


def calculate_size(size_image1, size_image2, homography,isaffine):

    # setup width and height
    (h1, w1) = size_image1[:2]
    (h2, w2) = size_image2[:2]

    # remap the coordinates of the projected image onto the panorama image
    # space
    top_left = np.dot(homography, np.asarray([0, 0, 1]))
    top_right = np.dot(homography, np.asarray([w2, 0, 1]))
    bottom_left = np.dot(homography, np.asarray([0, h2, 1]))
    bottom_right = np.dot(homography, np.asarray([w2, h2, 1]))

    if not isaffine:
    # normalize
        top_left = top_left / top_left[2]
        top_right = top_right / top_right[2]
        bottom_left = bottom_left / bottom_left[2]
        bottom_right = bottom_right / bottom_right[2]

    pano_left = int(min(top_left[0], bottom_left[0], 0))
    pano_right = int(max(top_right[0], bottom_right[0], w1))
    width_w = pano_right - pano_left

    pano_top = int(min(top_left[1], top_right[1], 0))
    pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
    height_h = pano_bottom - pano_top

    size = (width_w, height_h)

    # offset of first image relative to panorama
    offset_x = int(min(top_left[0], bottom_left[0], 0))
    offset_y = int(min(top_left[1], top_right[1], 0))
    offset = (-offset_x, -offset_y)

    return (size, offset)

#####################################################################

# Merges two images given the homography, new combined size for a
# combined mosiac/panorama and the translation offset vector between them


def merge_images(image1, image2, homography, size, offset):
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]


    panorama = np.zeros((size[1], size[0], 3), np.uint8)

    (ox, oy) = offset

    translation = np.array([[1.0, 0.0, ox],
                             [0, 1.0, oy],
                             [0.0, 0.0, 1.0]])

    homography = translation @ homography

    # draw the transformed image2 into the panorama
    #image2 = np.ones((size[1], size[0], 3), np.uint8)*255

    cv2.warpPerspective(image2, homography, size, panorama)

    # masking to work out overlaps

    mask_b = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    maskPano = np.zeros((size[1], size[0]), np.uint8)
    mask_a1 = np.ones((h2,w2), np.uint8)*255

    cv2.warpPerspective(mask_a1, homography, size, maskPano)

    panorama[maskPano<255] = 0
    maskPano[maskPano<255] = 0

    mask_a1 = maskPano[oy:h1 + oy, ox:ox + w1]


    #mask_a1[mask_a1 < 255] = 0


    a_and_b = cv2.bitwise_and(mask_a1, mask_b)
    overlap_area_mask = cv2.threshold(a_and_b, 0, 255, cv2.THRESH_BINARY)[1]

    a_nonoverlap_area_mask = cv2.threshold(mask_a1, 0, 255, cv2.THRESH_BINARY)[
        1] - overlap_area_mask
    b_nonoverlap_area_mask = cv2.threshold(mask_b, 0, 255, cv2.THRESH_BINARY)[
        1] - overlap_area_mask

    # previous part of panorama (before this frame) - only (part of image1
    # not covered by image2)

    ored = cv2.bitwise_or(panorama[oy:h1 +
                                   oy, ox:ox +
                                   w1], image1, mask=(b_nonoverlap_area_mask -
                                                      a_nonoverlap_area_mask))

    oredcorrect = cv2.subtract(ored, panorama[oy:h1 + oy, ox:ox + w1])

    # final composition

    panorama[oy:h1 + oy, ox:ox +
             w1] = cv2.add(panorama[oy:h1 + oy, ox:ox + w1], oredcorrect)

    # cv2.imshow("a", mask_a1)
    # cv2.imshow("b", mask_b)
    # cv2.imshow("and", a_and_b * 255)
    # cv2.imshow("over", overlap_area_mask)
    # cv2.imshow("a non over", a_nonoverlap_area_mask)
    # cv2.imshow("b non over", b_nonoverlap_area_mask)
    # cv2.imshow("minus", b_nonoverlap_area_mask - a_nonoverlap_area_mask)
    # cv2.imshow("panorama_final",panorama)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return panorama, homography, translation


def merge_costMaps(costMap1, costMap2, homography, size, offset):
    (h1, w1) = costMap1.shape[:2]
    (h2, w2) = costMap2.shape[:2]

    panorama = np.zeros((size[1], size[0]), np.float32)


    (ox, oy) = offset

    translation = np.array([[1.0, 0.0, ox],
                             [0, 1.0, oy],
                             [0.0, 0.0, 1.0]])

    homography = translation @ homography

    # draw the transformed image2 into the panorama
    #image2 = np.ones((size[1], size[0], 3), np.uint8)*255

    cv2.warpPerspective(costMap2, homography, size, panorama)

    # masking to work out overlaps

    mask_b = costMap1.copy()#cv2.cvtColor(costMap1, cv2.COLOR_RGB2GRAY)
    #mask_b[mask_b>0] = 255
    mask_b = convert2Img(mask_b)

    maskPano = np.zeros((size[1], size[0]), np.uint8)
    mask_a1 = np.ones((h2,w2), np.uint8)*255

    cv2.warpPerspective(mask_a1, homography, size, maskPano)

    panorama[maskPano<255] = 0
    maskPano[maskPano<255] = 0

    mask_a1 = maskPano[oy:h1 + oy, ox:ox + w1]


    #mask_a1[mask_a1 < 255] = 0


    a_and_b = cv2.bitwise_and(mask_a1, mask_b)
    overlap_area_mask = cv2.threshold(a_and_b, 0, 255, cv2.THRESH_BINARY)[1]

    a_nonoverlap_area_mask = cv2.threshold(mask_a1, 0, 255, cv2.THRESH_BINARY)[
        1] - overlap_area_mask
    b_nonoverlap_area_mask = cv2.threshold(mask_b, 0, 255, cv2.THRESH_BINARY)[
        1] - overlap_area_mask

    # previous part of panorama (before this frame) - only (part of image1
    # not covered by image2)


    ored = cv2.bitwise_or(panorama[oy:h1 +
                                   oy, ox:ox +
                                   w1], costMap1, mask=(b_nonoverlap_area_mask -
                                                        a_nonoverlap_area_mask))

    oredcorrect = cv2.subtract(ored, panorama[oy:h1 + oy, ox:ox + w1])
    oredcorrect[oredcorrect<0] = 0

    # final composition

    panorama[oy:h1 + oy, ox:ox +
             w1] = cv2.add(panorama[oy:h1 + oy, ox:ox + w1], oredcorrect)



    # cv2.imshow("a", mask_a1)
    # cv2.imshow("b", mask_b)
    # cv2.imshow("and", a_and_b * 255)
    # cv2.imshow("over", overlap_area_mask)
    # cv2.imshow("a non over", a_nonoverlap_area_mask)
    # cv2.imshow("b non over", b_nonoverlap_area_mask)
    # cv2.imshow("minus", b_nonoverlap_area_mask - a_nonoverlap_area_mask)
    # cv2.imshow("orer",convert2ImgHot(ored))
    # cv2.imshow("orer_corr",convert2ImgHot(oredcorrect))
    # cv2.imshow("panorama_final",convert2ImgHot(panorama))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return panorama, homography, translation

def convert2Img(arr,dsMax=400):
    arr = arr * (255/dsMax)
    arr = np.ceil(arr)
    return arr.astype(np.uint8)

def convert2ImgHot(arr):
    return cv2.applyColorMap(convert2Img(arr), cv2.COLORMAP_HOT)

# def merge_images(image1, image2, homography, size, offset,isAffine):
#     (h1, w1) = image1.shape[:2]
#     (h2, w2) = image2.shape[:2]
#
#     # image1 = image1.astype(np.float32)
#     # image2 = image2.astype(np.float32)
#     # panorama = np.zeros((size[1], size[0], 3), np.float32)
#     panorama = np.zeros((size[1], size[0], 3), np.uint8)
#
#     (ox, oy) = offset
#
#
#     translation = np.array([[1.0, 0.0, ox],
#                             [0, 1.0, oy],
#                             [0.0, 0.0, 1.0]])
#
#     homography = translation @ homography
#
#     imgShape = image2.shape
#     corners = [[0, 0], [0, imgShape[0]], [imgShape[1], 0], [imgShape[1], imgShape[0]]]
#     corners = np.asarray(corners)
#     cWarp = cv2.perspectiveTransform(corners.astype(np.float32).reshape(-1, 1, 2), homography)
#     cWarp = cWarp.astype(int)
#
#     # draw the transformed image2 into the panorama
#
#
#     cv2.warpPerspective(image2, homography, size, panorama)
#     # cv2.imshow("panorama", panorama)
#     # cv2.waitKey(0)
#     warpedResImg = cv2.warpPerspective(image2, translation, size)
#
#     #masking to work out overlaps
#
#     mask_a = cv2.cvtColor(panorama[oy:h1 + oy, ox:ox + w1], cv2.COLOR_RGB2GRAY)
#     mask_b = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
#     a_and_b = cv2.bitwise_and(mask_a, mask_b)
#     overlap_area_mask = cv2.threshold(a_and_b, 1, 255, cv2.THRESH_BINARY)[1]
#
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#     # overlap_area_mask = cv2.morphologyEx(overlap_area_mask, cv2.MORPH_ERODE, kernel)
#
#     a_nonoverlap_area_mask = cv2.threshold(mask_a, 1, 255, cv2.THRESH_BINARY)[
#         1] - overlap_area_mask
#
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#     # a_nonoverlap_area_mask = cv2.morphologyEx(a_nonoverlap_area_mask, cv2.MORPH_ERODE, kernel)
#
#     b_nonoverlap_area_mask = cv2.threshold(mask_b, 1, 255, cv2.THRESH_BINARY)[
#         1] - overlap_area_mask
#
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     # b_nonoverlap_area_mask = cv2.morphologyEx(b_nonoverlap_area_mask, cv2.MORPH_ERODE, kernel)
#
#     # previous part of panorama (before this frame) - only (part of image1
#     # not covered by image2)
#     # image1 = image1.astype(np.uint8)
#     # image2 = image2.astype(np.uint8)
#     # panorama = panorama.astype(np.uint8)
#
#     ored = cv2.bitwise_or(panorama[oy:h1 +
#                                    oy, ox:ox +
#                                    w1], image1, mask=(b_nonoverlap_area_mask.astype(np.uint8) -
#                                                       a_nonoverlap_area_mask.astype(np.uint8)))
#
#     oredcorrect = cv2.subtract(ored, panorama[oy:h1 + oy, ox:ox + w1])
#
#
#     # final composition
#     # mask2 = cv2.threshold(panorama, 0, 255, cv2.THRESH_BINARY)[1]
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, kernel)
#     # panorama[mask2 == 0] = 0
#
#     panorama[oy:h1 + oy, ox:ox + w1] = cv2.add(panorama[oy:h1 + oy, ox:ox + w1].astype(np.float32), oredcorrect.astype(np.float32))
#
#     #fix the line thingy
#
#     maskBoi = np.zeros(panorama.shape[:2],dtype=np.uint8)
#
#
#     # maskBoi[cockSucker>255] = 255
#
#     cv2.line(maskBoi, cWarp[0][0], cWarp[1][0], (255, 255, 255), 3, 8)
#     cv2.line(maskBoi, cWarp[1][0], cWarp[3][0], (255, 255, 255), 3, 8)
#     cv2.line(maskBoi, cWarp[3][0], cWarp[2][0], (255, 255, 255), 3, 8)
#     cv2.line(maskBoi, cWarp[2][0], cWarp[0][0], (255, 255, 255), 3, 8)
#
#
#     #
#     # #find anything in the mask where the panno is being dumb
#     # maskBoi[np.bitwise_and(np.all(panorama==(30,30,30),axis=2),maskBoi==155)] = 255
#     # maskBoi = cv2.threshold(maskBoi,200,255,cv2.THRESH_BINARY)[1]
#     #
#     #maskBoi[np.bitwise_and(maskBoi ==255,np.all(cv2.threshold(image1,1,255,cv2.THRESH_BINARY)[1])==255)] = 0
#
#     maskBoiI1 = copy.deepcopy(maskBoi)
#     maskBoiI1 = maskBoiI1[:image1.shape[0],:image1.shape[1]]
#
#     fuckOff = np.zeros(panorama.shape[:2],dtype=np.uint8)
#     fuckOff[:image1.shape[0],:image1.shape[1]] = 1
#     maskBoi = maskBoi * fuckOff
#
#     pc = copy.deepcopy(panorama)
#     # s = 15
#     # kernel = np.ones((s, s), np.float32) / s**2
#
#     kernel = np.array([[1,1,1,1,1,1,1],[1,1,1,0,1,1,1],[1,1,0,0,0,1,1],[1,0,0,0,0,0,1],[1,1,0,0,0,1,1],[1,1,1,0,1,1,1],[1,1,1,1,1,1,1]])/(49-13)
#     blur = cv2.filter2D(pc,-1,kernel)
#
#     panorama[maskBoi>1] = blur[maskBoi>1]
#
#     # panorama[maskBoi==255] = image1[maskBoiI1==255]
#
#     # panorama[maskBoi==255] = ((image1[maskBoiI1==255].astype(np.float32) + panorama[maskBoi==255].astype(np.float32))/2).astype(np.uint8)
#     #panorama[panorama==0] = pc[panorama==0]
#     # indices = nd.distance_transform_edt(maskBoi==255, return_distances=False, return_indices=True)
#     # panorama = panorama[tuple(indices)]
#
#     #cv2.warpPerspective(np.ones(image2.shape[:2],dtype=np.uint8)*130, homography, size, maskBoi)
#
#     # cv2.imshow("square mask", maskBoi.astype(np.uint8))
#     # cv2.imshow("panorama", panorama)
#     # cv2.imshow("ord", image1)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#
#     # panorama = np.where(panorama != 0, panorama, warpedResImg)
#
#
#     return panorama.astype(np.uint8), homography, translation

#####################################################################