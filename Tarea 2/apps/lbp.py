import numpy as np
from utils.utils import getThresholdImgs, printImg
from skimage.feature import local_binary_pattern


img = np.array([[4, 6, 9, 6, 4, 6],
               [9, 6, 4, 9, 9, 9],
               [9, 6, 2, 2, 9, 2],
               [10, 10, 10, 10, 10, 10]], dtype=np.uint8)

patterns = local_binary_pattern(img, 8, 1, 'default')

print(patterns)
