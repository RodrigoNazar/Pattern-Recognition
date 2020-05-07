import numpy as np
from utils.utils import getThresholdImgs, printImg
from skimage.feature import local_binary_pattern

n_bins = 123

img = np.array([[4, 6, 9, 6, 4, 6],
               [9, 6, 4, 9, 9, 9],
               [9, 6, 2, 2, 9, 2],
               [10, 10, 10, 10, 10, 10]], dtype=np.uint8)

patterns = local_binary_pattern(img, P=8, R=1, method='default')
hist, _ = np.histogram(patterns, density=True, bins=n_bins, range=(0, n_bins))


print(patterns)
print(hist)
