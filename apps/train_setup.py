
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
import os

from utils.utils import (getThresholdImgs, printImg, isInside, segmentate,
                         extractSubmatrix)
import utils.huMoments as hu


def setupTrainImg(img_path):

    print('\nObteniendo características del archivo', img_path, '...')

    img =  cv2.imread(img_path)

    th_R, th_G, th_B = getThresholdImgs(img)

    # img = 255*((th_R > 127) | (th_G > 127) | (th_B > 127))

    rects = segmentate(th_R)

    data = {
        'img': img_path,
        'objects': []
    }

    for (x, y, w, h) in rects:

        # Se despliega la letra
        current_img = extractSubmatrix(th_R, (x, y), (x+w, y+h))
        printImg(current_img, 1, 200)

        # El usuario indica que letra es
        letter = input('\n¿Que letra era esa?\n\t> ')

        # Se calculan los momentos de Hu y se agrega a los datos
        phi1, phi2, phi3, phi4, phi5, phi6, phi7 = hu.huMoments(current_img)
        data['objects'].append(
            {
                'coord': [x, y, x+w, y+h],
                'letter': letter,
                'hu-moments': {
                    'phi1': phi1,
                    'phi2': phi2,
                    'phi3': phi3,
                    'phi4': phi4,
                    'phi5': phi5,
                    'phi6': phi6,
                    'phi7': phi7
                }
             }
        )

    # Finalmente se guarda la información de las letras
    data_path = 'data/' + img_path[4:-4] + '_data.json'
    print('\nGuardando los datos en: ', data_path)
    with open(data_path, 'a') as json_file:
        json.dump(data, json_file, indent=2)


def train_setup(*args):

    # Se obtienen las características de las imágenes
    for pic in args:
        setupTrainImg(pic)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t01_path', type=str, default='img/Training_01.png',
                        help="Dirección de la imagen a procesar")
    parser.add_argument('--t02_path', type=str, default='img/Training_02.png',
                        help="Dirección de la imagen a procesar")
    cmd_args = parser.parse_args()
    train_setup(cmd_args.t01_path, cmd_args.t02_path)
