
import cv2
import json

import utils.huMoments as hu
from utils.utils import getThresholdImgs, printImg
from apps.testing import getDataFiles, knn, huDifference


def reconocedor(X):
    '''
    % Recibe una letra en formato binario
    % inputs
    % X ~ es un numpy.array de mxn en formato unit8. con 1 si es letra y 0 si no es

    % output
    % letra ~ valor entre 1 y 5 indicando la letra clasificada.
           % 1. a
           % 2. s
           % 3. d
           % 4. f
           % 5. g

    % Ejemplo: reconecedor(A) = 1 , donde A representa una matriz con la letra A.
    '''
    corresponds = {
        'A': 1,
        'S': 2,
        'D': 3,
        'F': 4,
        'G': 5
    }

    # Obtenemos los momentos de hu de X
    phi1, phi2, phi3, phi4, phi5, phi6, phi7 = hu.huMoments(X)
    testing_json = {
        'phi1': phi1,
        'phi2': phi2,
        'phi3': phi3,
        'phi4': phi4,
        'phi5': phi5,
        'phi6': phi6,
        'phi7': phi7
    }

    # Ahora, les aplicamos una normalización Min Max
    with open(f'data/testing_data.json', 'r') as json_file:
        json_data = json.load(json_file)

    huMomentsUsed = ('phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7')

    # Obtenemos los valores de cada uno de los momentos calculados
    moments_data = {mom: [] for mom in huMomentsUsed}

    for letter in json_data:
        for data in json_data[letter]:
            for huElem in data['data']:
                moments_data[huElem].append(data['data'][huElem])

    # Creamos un diccionario con los valores máximos y mínimos de cada momento
    normalizations = {}
    for moment in moments_data:
        min_val, max_val = min(moments_data[moment]), max(moments_data[moment])
        # print(moment, min_val, max_val)
        normalizations[moment] = {'min': min_val,
                                  'max': max_val}

    # Aplicamos la normalización de los momentos de X
    for moment in testing_json:
        min_val, max_val = normalizations[moment]['min'], normalizations[moment]['max']
        testing_json[moment] = (testing_json[moment] - min_val)/(max_val-min_val)

    # Ahora procedemos a comparar los momentos normalizados de X con los del
    # training
    training_data, _ = getDataFiles()
    differences = []

    for letter in training_data:
        for training_json in training_data[letter]:
            # Obtenemos la diferencia y la guardamos
            difference = huDifference(testing_json,
                           training_json['data'],
                           huMomentsUsed)

            dif_data = {
                'letter': letter,
                'difference': difference
            }

            differences.append(dif_data)

    # Obtenemos el resultado consultando a los 5 vecinos más cercanos
    result = knn(differences, 5)

    return corresponds[result]


def reconocedorTest():
    # Letra A
    A = cv2.imread('img/testing/A/A_22.png')
    th_A, _, _ = getThresholdImgs(A)

    # Letra S
    S = cv2.imread('img/testing/S/S_0.png')
    th_S, _, _ = getThresholdImgs(S)

    # Letra D
    D = cv2.imread('img/testing/D/D_3.png')
    th_D, _, _ = getThresholdImgs(D)

    # Letra F
    F = cv2.imread('img/testing/F/F_32.png')
    th_F, _, _ = getThresholdImgs(F)

    # Letra G
    G = cv2.imread('img/testing/G/G_26.png')
    th_G, _, _ = getThresholdImgs(G)

    print('Reconocedor(A) =', reconocedor(th_A))
    print('Reconocedor(S) =', reconocedor(th_S))
    print('Reconocedor(D) =', reconocedor(th_D))
    print('Reconocedor(F) =', reconocedor(th_F))
    print('Reconocedor(G) =', reconocedor(th_G))


if __name__ == '__main__':
    reconocedorTest()
