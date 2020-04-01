'''
        UTILS:

Saca bordes:
cv2.Canny(img,100,200)

Dibuja rectangulos:
cv2.rectangle(img, (500, 500), (1000, 1000), (255, 0, 0), 50)

Muestra 3 imagenes:
plt.figure(figsize=(75, 75))
plt.subplot(131), plt.imshow(th_R, cmap='gray')
plt.title('R Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(th_G, cmap='gray')
plt.title('G Image'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(th_B, cmap='gray')
plt.title('B Image'), plt.xticks([]), plt.yticks([])

plt.show()
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import json


def getThresholdImgs(img):
    '''
    Compute the threshold image of the three canals
    :params:
    img

    :returns:
    th_R, th_G, th_B
    '''

    img_R, img_G, img_B = cv2.split(img)

    _, th_R = cv2.threshold(img_R, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, th_G = cv2.threshold(img_G, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, th_B = cv2.threshold(img_B, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return th_R, th_G, th_B


def printImg(img, factor, time):
    '''
    Prints an image and resize it by the factor
    '''
    if len(img.shape) == 2:
        x, y = img.shape
    elif len(img.shape) == 3:
        x, y, _ = img.shape

    if factor != 0:
        x = int(x/factor)
        y = int(y/factor)

    imgr = cv2.resize(img, (y, x))

    cv2.imshow('Training_01', imgr)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def isInside(out1, out2, in1, in2):

    x = out1[0] < in1[0] < in2[0] < out2[0]
    y = out1[1] < in1[1] < in2[1] < out2[1]
    return x and y


def segmentate(img):
    mser = cv2.MSER_create()

    regions, rects = mser.detectRegions(img)

    if isinstance(rects, list):
        rects = [i for i in rects if (i[0] != 1 and i[1] != 1)
                                          and (i[0] != 0 and i[1] != 0)]
    else:
        rects = [i for i in rects.tolist() if (i[0] != 1 and i[1] != 1)
                                          and (i[0] != 0 and i[1] != 0)]
    insideRects = []

    # Filters rects that are inside of others
    for (x1, y1, w1, h1) in rects:
        for (x2, y2, w2, h2) in rects:
            if isInside((x1, y1), (x1+w1, y1+h1), (x2, y2), (x2+w2, y2+h2)):
                insideRects.append([x2, y2, w2, h2])

    return [i for i in rects if i not in insideRects]


def extractSubmatrix(matrix, upperPoint, downerPoint):
    return np.array([ x[upperPoint[0]:downerPoint[0]]
            for x in matrix[upperPoint[1]:downerPoint[1]]])
