import numpy as np

from utils.utils import printImg

def momentRS(img, r, s):
    whitePixels = np.argwhere(img == 255)

    m = 0

    for pixl in whitePixels:
        y = pixl[0]
        x = pixl[1]
        m += (x**r)*(y**s)

    return m


def centroid(img):
    x = momentRS(img, 1, 0) / momentRS(img, 0, 0)

    y = momentRS(img, 0, 1) / momentRS(img, 0, 0)

    return x, y


def uMomentRS(img, r, s):
    x_cent, y_cent = centroid(img, r, s)

    whitePixels = np.argwhere(img == 255)

    u = 0

    for pixl in whitePixels:
        y = pixl[0]
        x = pixl[1]
        u += ((x - x_cent)**r)*((y - y_cent)**s)

    return u


def etaMomentRS(img, r, s):
    t = (r + s)/2 + 1
    return uMomentRS(img, r, s) / (uMomentRS(img, 0, 0)**t)


def huMoments(img):

    # Etas
    eta02 = etaMomentRS(img, 0, 2)
    eta03 = etaMomentRS(img, 0, 3)
    eta11 = etaMomentRS(img, 1, 1)
    eta12 = etaMomentRS(img, 1, 2)
    eta20 = etaMomentRS(img, 2, 0)
    eta21 = etaMomentRS(img, 2, 1)
    eta30 = etaMomentRS(img, 3, 0)

    # Momentos
    phi1 = eta20 + eta02
    phi2 = (eta20 - eta02)**2 + 4*eta11**2
    phi3 = (eta30 - 3*eta12)**2 + (3*eta21 - eta03)**2
    phi4 = (eta30 + eta12)**2 + (eta21 + eta03)**2
    phi5 = (eta30 - 3*eta12)*(eta30 + eta12)*((eta30 + eta12)**2 - 3*(eta21 + eta03)**2)
            + (3*eta21 - eta03)*(eta21 + eta03)*(3*(eta30 + eta12)**2 - (eta21 + eta03)**2)
    phi6 = (eta20 - eta02)*((eta30 + eta12)**2 - (eta21 + eta03)**2) +
            4*eta11*(eta30 + eta12)*(eta21 + eta03)
    phi7 = (3*eta21 - eta03)*(eta30 + eta12)*((eta30 + eta12)**2 - 3*(eta21 + eta03)**2)
            - (eta30 - 3*eta12)*(eta21 + eta03)*(3*(eta30 + eta12)**2 - (eta21 + eta03)**2)

    return phi1, phi2, phi3, phi4, phi5, phi6, phi7
