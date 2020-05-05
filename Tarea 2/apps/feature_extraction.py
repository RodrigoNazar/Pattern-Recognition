mod = '''
1. Extracción de Características
'''

import os
import cv2
from skimage.feature import local_binary_pattern
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import getThresholdImgs, printImg

classes = ['rayada', 'no_rayada']


def FeatureExtractor(training_path='img/training',
                     testing_path='img/testing',
                     classes=classes):

        # Features of training images
        for _class in classes:
            dir_path = os.listdir(os.path.join(training_path, _class))
            for img in dir_path:

                img_path = os.path.join(training_path, _class, img)
                img = cv2.imread(img_path)

                # Obtenemos los umbrales de cada canal
                th_R, th_G, th_B = getThresholdImgs(img)

                

                break


if __name__ == '__main__':
    print(mod)
