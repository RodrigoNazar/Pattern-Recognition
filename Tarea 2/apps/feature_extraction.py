mod = '''
1. Extracción de Características
'''

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

from utils.utils import getThresholdImgs, printImg
from utils.gabor import GaborFeatures
from utils.haralick import HaralickFeatures
from utils.lbp import LBPFeatures

from datetime import datetime


classes = ['rayada', 'no_rayada']
feature_names = []
for channel in ('Red ', 'Green ', 'Blue '):
    # LBP features
    feature_names += [channel + 'lpb' + str(i) for i in range(59)]

    # Haralick features
    for feature in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']:
        for dir in range(8):
            feature_names.append(channel + feature + f' dir{dir}')

    # Gabor features
    gabor = [channel + 'gabor_func' + str(i) for i in range(4)]
    for feature in gabor:
        feature_names.append(feature + ' mean')
        feature_names.append(feature + ' var')
        feature_names.append(feature + ' sum')


def FeatureComputator(img_path):
    img = cv2.imread(img_path)

    # Obtenemos los umbrales de cada canal
    th_R, th_G, th_B = getThresholdImgs(img)


    # FEATURE EXTRACTION

    # LBP features
    #   > 59 features por canal
    lbp_R = LBPFeatures(th_R)
    lbp_G = LBPFeatures(th_G)
    lbp_B = LBPFeatures(th_B)


    # Haralick features
    #   > 48 features por canal
    haralick_R = HaralickFeatures(th_R)
    haralick_G = HaralickFeatures(th_G)
    haralick_B = HaralickFeatures(th_B)


    # Gabor features
    #   > 12 features por canal
    gabor_R = GaborFeatures(th_R)
    gabor_G = GaborFeatures(th_G)
    gabor_B = GaborFeatures(th_B)


    # Features por canal
    features_R = np.concatenate((lbp_R, haralick_R, gabor_R))
    features_G = np.concatenate((lbp_G, haralick_G, gabor_G))
    features_B = np.concatenate((lbp_B, haralick_B, gabor_B))


    # Vector final de features, de lago (59 + 48 + 12)*3 = 357
    return np.concatenate((features_R, features_G, features_B)).tolist()



def FeatureExtractor(training_path='img/training',
                     testing_path='img/testing',
                     classes=classes):

        print('\nObteniendo las características ...')

        # Si ya se calcularon las características, sólo las consultamos
        existsData = os.listdir('data/')
        if 'paredes_data.json' in existsData:
            print('Se encontraron características ya calculadas!')
            with open(os.path.join('data/', existsData[0]), 'r') as file:
                return json.loads(file.read())

        # Si no se han calculado las características, lo hacemos
        print('Calculando las características de las imágenes:\n')
        data = {
            'feature_names': feature_names,
            'feature_values_train': [],
            'labels_train': [],
            'feature_values_test': [],
            'labels_test': [],

        }

        start = datetime.now()

        for _class in classes:

            i = 0

            # Características de las imágenes de training
            dir_path = os.listdir(os.path.join(training_path, _class))
            for img in dir_path:

                img_path = os.path.join(training_path, _class, img)
                print(f'Procesando {img_path}, imagen número {i}')

                features = FeatureComputator(img_path)
                label = 1 if _class == 'rayada' else 2

                data['feature_values_train'].append(features)
                data['labels_train'].append(label)
                i += 1

            # Características de las imágenes de testing
            dir_path = os.listdir(os.path.join(testing_path, _class))
            for img in dir_path:

                img_path = os.path.join(testing_path, _class, img)
                print(f'Procesando {img_path}, imagen número {i}')

                features = FeatureComputator(img_path)
                label = 1 if _class == 'rayada' else 2

                data['feature_values_test'].append(features)
                data['labels_test'].append(label)

                i += 1

        # Finalmente guardamos los datos en un archivo
        with open('data/paredes_data.json', 'w') as json_file:
            json_file.write(json.dumps(data))

        print('Tiempo tomado por la extracción: ', datetime.now() - start)

        return data


if __name__ == '__main__':
    print(mod)
