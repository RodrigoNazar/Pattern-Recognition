import cv2 as cv


def main():
    img = cv.imread('pics/Training_01.png')
    img = cv.resize(img, (960, 540))
    cv.imshow('Training_01', img)
    cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    main()
