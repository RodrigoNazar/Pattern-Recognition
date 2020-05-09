mod = '''
1. Extracción de Características
'''

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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



def FeatureExtractor(training_path='img/training',
                     testing_path='img/testing',
                     classes=classes):

        # Features of training images
        for _class in classes:
            dir_path = os.listdir(os.path.join(training_path, _class))
            for img in dir_path:
                start = datetime.now()

                img_path = os.path.join(training_path, _class, img)
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
                features = np.concatenate((features_R, features_G, features_B))

                print('datetime', datetime.now()- start)



                break


if __name__ == '__main__':
    print(mod)
