
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import json

from utils.utils import (getThresholdImgs, printImg, isInside, segmentate,
                         extractSubmatrix)
import utils.huMoments as hu


def setupTrainImg(img):

    th_R, th_G, th_B = getThresholdImgs(img)

    # img = 255*((th_R > 127) | (th_G > 127) | (th_B > 127))

    rects = segmentate(th_R)

    cache = []
    for (x, y, w, h) in rects:
        printImg(extractSubmatrix(th_R, (x, y), (x+w, y+h)), 1, 1000)

        letter = input()
        cache.append(
            {
                'coord': [x, y, w, h],
                'letter': letter
             }
        )

    with open('cache/training_data.json', 'w') as json_file:
        json.dump(cache, json_file)


def main(t01_path, t02_path):
    train01 = cv2.imread(t01_path)
    train02 = cv2.imread(t02_path)

    th_R, th_G, th_B = getThresholdImgs(train01)

    x, y = hu.centroid(th_R)

    print('x =', x, 'y =', y)

    printImg(th_R, 1, 2*6000)

    # setupTrainImg(train01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t01_path', type=str, default='img/prueba2.png',
                        help="Dirección de la imagen a procesar")
    parser.add_argument('--t02_path', type=str, default='img/Training_02.png',
                        help="Dirección de la imagen a procesar")
    cmd_args = parser.parse_args()
    main(cmd_args.t01_path, cmd_args.t02_path)
