import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt


'''
Módulo inspirado en el código expuesto en:
https://scikit-image.org/docs/0.12.x/auto_examples/features_detection/plot_gabor.html
'''


def GaborFunctions(sigma, frequency, angles):
    kernels = []
    for theta in range(angles):
        theta = theta / angles * np.pi
        kernel = np.real(gabor_kernel(frequency, theta=theta,
                                      sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel)

    return kernels


def GaborFeatures(img, sigma=5, frequency=.25, angles=4):
    kernels = GaborFunctions(sigma, frequency, angles)

    feats = np.zeros((3*len(kernels)), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(img, kernel, mode='wrap')
        feats[3*k] = filtered.mean()
        feats[3*k + 1] = filtered.var()
        feats[3*k + 2] = filtered.sum()

    return feats
