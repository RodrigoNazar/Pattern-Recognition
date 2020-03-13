import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def printImg(img, factor):
    if len(img.shape) == 2:
        x, y = img.shape
    elif len(img.shape) == 3:
        x, y, _ = img.shape

    x = int(x/factor)
    y = int(y/factor)

    imgr = cv.resize(img, (y, x))

    cv.imshow('Training_01', imgr)
    cv.waitKey(3000)
    cv.destroyAllWindows()


def main():
    img = cv.imread('pics/Training_01.png')

    # img = cv.rectangle(img, (500, 500), (1000, 1000), (255, 0, 0), 50)

    img_R, img_G, img_B = cv.split(img)

    # _, th_R = cv.threshold(img_R, 127, 255, cv.THRESH_BINARY)

    print(img_B.equal)

    # printImg(th_R, 4)

    # hist, bins = np.histogram(img_R, bins=3)
    # plt.hist(th_R)
    # plt.show()

    # printImg(img, 4)


if __name__ == '__main__':
    main()
