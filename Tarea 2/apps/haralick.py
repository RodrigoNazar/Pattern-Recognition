import numpy as np
from utils.utils import getThresholdImgs, printImg
from skimage.feature import greycomatrix


img = np.array([[1, 0, 1, 2],
                [1, 1, 0, 0],
                [0, 2, 1, 0],
                [0, 2, 0, 2]], dtype=np.uint8)

result = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)

print(result)
print(result.shape, type(result))
