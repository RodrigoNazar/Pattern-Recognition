
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
import os


def setup(*pics):

    for pic in pics:
         pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr1_path', type=str, default='img/Testing.png',
                        help="Dirección de la imagen a procesar")
    cmd_args = parser.parse_args()
    testImages(cmd_args.tr1_path)
