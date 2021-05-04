import random
import cv2 as cv
import numpy as np

def shift(x,y,ratio= 0.0):
    #https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
    assert ratio < 1 and ratio > 0, 'Value should be less than 1 and greater than 0'

    ratioh = random.uniform(-ratio, ratio)
    ratiov = random.uniform(-ratio, ratio)

    h, w = x.shape[:2]

    moveh = w * ratioh
    movev = w * ratiov

    if ratioh > 0:
        x = x[:, :int(w - moveh), :]
    if ratioh < 0:
        x = x[:, int(-1 * moveh):, :]

    if ratiov > 0:
        x = x[:int(h - movev), :, :]
    if ratiov < 0:
        x = x[int(-1 * movev):, :, :]

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)

    if ratioh > 0:
        y = y[:, :int(w - moveh), :]
    if ratioh < 0:
        y = y[:, int(-1 * moveh):, :]

    if ratiov > 0:
        y = y[:int(h - movev), :, :]
    if ratiov < 0:
        y = y[int(-1 * movev):, :, :]



    x = fill(x, h, w)
    y = fill(y, h, w)

    return x, np.squeeze(y)

def rotate(x,y,angle):

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)

    angle = int(random.uniform(-angle, angle))
    h, w = x.shape[:2]

    A = cv.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    x = cv.warpAffine(x, A, (w, h))
    y = cv.warpAffine(y, A, (w, h))

    return x, np.squeeze(y)

def flip(x,y):
    hf = random.randint(0,1)
    vf = random.randint(0,1)

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)

    if hf == 1:
        x = cv.flip(x,1)
        y = cv.flip(y,1)

    if vf == 1:
        x = cv.flip(x,0)
        y = cv.flip(y,0)

    return x, np.squeeze(y)

def zoom(x, y, range):
    #https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
    assert range < 1 and range > 0, 'Value should be less than 1 and greater than 0'
    range = random.uniform(range, 1)

    if len(y.shape) == 2:
        y  = np.expand_dims(y,axis=2)


    h, w = x.shape[:2]
    h_taken = int(range * h)
    w_taken = int(range * w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    x = x[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    y = y[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    x = fill(x, h, w)
    y = fill(y, h, w)

    return x, np.squeeze(y)

def fill(img, h, w):
    #https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
    img = cv.resize(img, (h, w), cv.INTER_NEAREST) #INTER_CUBIC
    return img

def noise(x,y,cut):
    c = int(255*cut)
    n1 = np.random.randint(0, 255, (256, 256, 3))
    n2 = np.random.randint(0, 255, (256, 256, 3))

    n1[n1> c] = 0
    n2[n2 < 255-c] = 0
    x = x + (n1+n2)
    x[x> 255] = 255
    return x, y
