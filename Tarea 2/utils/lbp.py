import numpy as np
from utils.utils import getThresholdImgs, printImg
from skimage.feature import local_binary_pattern


def LBPFeatures(img):
    patterns = local_binary_pattern(img, P=8, R=1, method='nri_uniform')

    n_bins = int(patterns.max() + 1)
    hist, _ = np.histogram(patterns.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    return hist
