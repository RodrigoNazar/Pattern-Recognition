import numpy as np
from utils.utils import getThresholdImgs, printImg
from skimage.feature import greycomatrix, greycoprops


def HaralickFeatures(img, directions=8):
    angles = [i*np.pi/directions for i in range(directions)]
    result = greycomatrix(img, [1], angles, symmetric=True, levels=img.max()+1)

    # features = {
    #     'contrast': greycoprops(result, 'contrast'),
    #     'dissimilarity': greycoprops(result, 'dissimilarity'),
    #     'homogeneity': greycoprops(result, 'homogeneity'),
    #     'ASM': greycoprops(result, 'ASM'),
    #     'energy': greycoprops(result, 'energy'),
    #     'correlation': greycoprops(result, 'correlation'),
    # }
    contrast = greycoprops(result, 'contrast')[0]
    dissimilarity = greycoprops(result, 'dissimilarity')[0]
    homogeneity = greycoprops(result, 'homogeneity')[0]
    ASM = greycoprops(result, 'ASM')[0]
    energy = greycoprops(result, 'energy')[0]
    correlation = greycoprops(result, 'correlation')[0]

    # print('contrast', contrast.shape)
    # print('dissimilarity', dissimilarity.shape)
    # print('homogeneity', homogeneity.shape)
    # print('ASM', ASM.shape)
    # print('energy', energy.shape)
    # print('correlation', correlation.shape)

    return np.concatenate((contrast, dissimilarity, homogeneity, ASM, energy, correlation))
