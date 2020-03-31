'''
        UTILS:

Saca bordes:
cv2.Canny(img,100,200)

Dibuja rectangulos:
cv.rectangle(img, (500, 500), (1000, 1000), (255, 0, 0), 50)

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


import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import argparse


def getThresholdImgs(img):
    '''
    Compute the threshold image of the three canals
    :params:
    img

    :returns:
    th_R, th_G, th_B
    '''

    img_R, img_G, img_B = cv.split(img)

    _, th_R = cv.threshold(img_R, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, th_G = cv.threshold(img_G, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, th_B = cv.threshold(img_B, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    return th_R, th_G, th_B


def printImg(img, factor):
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

    imgr = cv.resize(img, (y, x))

    cv.imshow('Training_01', imgr)
    cv.waitKey(6000)
    cv.destroyAllWindows()


def isInside(out1, out2, in1, in2):

    x = out1[0] < in1[0] < in2[0] < out2[0]
    y = out1[1] < in1[1] < in2[1] < out2[1]

    # print(f'\n out1, out2, in1, in2: {out1}, {out2}, {in1}, {in2}')
    # print(f'x: {out1[0]} < {in1[0]} < {in2[0]} < {out2[0]} {x}')
    # print(f'y: {out1[1]} < {in1[1]} < {in2[1]} < {out2[1]} {y}')

    return x and y


def main(img_path):
    img = cv.imread(img_path)
    h, w = img.shape[:2]

    th_R, th_G, th_B = getThresholdImgs(img)

    # image_size = h*w
    mser = cv.MSER_create()

    print(type(mser))
    # mser.setMaxArea(int(image_size/2))
    # mser.setMinArea(10)

    regions, rects = mser.detectRegions(th_R)

    rects = [i for i in rects.tolist() if (i[0] != 1 and i[1] != 1)
                                      and (i[0] != 0 and i[1] != 0)]
    insideRects = []

    # With the rects you can e.g. crop the letters
    for (x1, y1, w1, h1) in rects:
        for (x2, y2, w2, h2) in rects:
            if isInside((x1, y1), (x1+w1, y1+h1), (x2, y2), (x2+w2, y2+h2)):
                insideRects.append([x2, y2, w2, h2])

    rects = [i for i in rects if i not in insideRects]

    for (x, y, w, h) in rects:
        cv.rectangle(th_R, (x, y), (x+w, y+h), color=(128, 128, 0), thickness=4)

    printImg(th_R, 2)

    # edges = cv.Canny(th_R, 20, 30)
    #
    # printImg(edges, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True,
                        help="Dirección de la imagen a procesar")
    cmd_args = parser.parse_args()
    main(cmd_args.img_path)
