
import cv2
from matplotlib import pyplot as plt
from functools import reduce
import numpy as np
import argparse
import json
import os
import itertools

from apps.statistics import successPercentage


def huDiference(hu1, hu2, moments):
    diff = [(hu1[moment], hu2[moment]) for moment in moments]

    return sum(map(lambda x: (x[0] - x[1])**2, diff))**.5


def knn(list, neighbors):
    list = sorted(list, key=lambda i: i['difference'])[:neighbors]
    rep = [letter['letter'] for letter in list]
    rep = {rep.count(i): i for i in rep}
    maximum = max([i for i in rep.keys()])
    rep = rep[maximum]

    return rep


def getDataFiles():

    # Obtenemos las features de las imágenes de training
    training_file = [file for file in os.listdir('data/') if 'training' in file][0]

    with open(f'data/{training_file}', 'r') as json_file:
        training_data = json.load(json_file)

    # Obtenemos las features de las imágenes de testing
    testing_file = [file for file in os.listdir('data/') if 'testing' in file][0]

    with open(f'data/{testing_file}', 'r') as json_file:
        testing_data = json.load(json_file)

    return training_data, testing_data


def testing(huMomentsUsed):

    training_data, testing_data = getDataFiles()

    results = []

    '''
    En los siguientes for tendremos:
    letter in ('A', 'S', 'D', 'F', 'G')
        -> La respuesta del test

    testing_json json
        -> Objeto json con la data de imagen testing que estamos
           analizando. Tiene la forma {
                'file': 'img/testing/G/G_12.png',
                'data': {'phi1': 0.24402917149737346,
                         'phi2': 4.271809180755498e-05,
                         'phi3': 0.00011388421406912314,
                         'phi4': 0.00012192698237651798,
                         'phi5': 5.537608079641481e-09,
                         'phi6': 1.415076616137105e-07,
                         'phi7': -1.3257459603662303e-08}
           }


    tLetter in ('A', 'S', 'D', 'F', 'G')
        -> Itera en las posibles letras de la solucion

    training_json json
        -> Objeto json con la data de imagen training que
           estamos analizando. Tiene la misma forma que testing_json.
    '''
    for letter in testing_data:
        for testing_json in testing_data[letter]:
            result_object = {
                'test_file': testing_json['file'],
                'answer': letter
            }
            differences = []
            for tLetter in training_data:
                # Hacemos la comparación con cada letra del training
                for training_json in training_data[tLetter]:
                    difference = huDiference(testing_json['data'],
                                   training_json['data'],
                                   huMomentsUsed)

                    dif_data = {
                        'letter': tLetter,
                        'difference': difference
                    }

                    differences.append(dif_data)



            # Obtenemos el resultado consultando a los 5 vecinos más cercanos
            result = knn(differences, 5)

            result_object['output'] = result

            results.append(result_object)

    # Ahora, tenemos un objeto JSON con cada letra del test con su respectivo
    # representante obtenido por knn de 5 vecinos
    return results


def huCombinationTest():

    print('\n3) Ejecutando el algoritmo de reconocimiento de ', end='')
    print('caracteres, con distintas combinaciones de momentos de hu...')

    # Momentos de hu
    huMomentsUsed = ('phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7')

    results = []

    # Obtenemos todas las posibles combinaciones entre los distintos momentos
    for length in range(len(huMomentsUsed)+1):
        for subset in itertools.combinations(huMomentsUsed, length):
            if subset: # Elimina la posibilidad vacía

                results.append({
                    'momentos': subset,
                    'successPercentage': successPercentage(testing(subset))
                })

    result = sorted(results, key=lambda i: i['successPercentage'], reverse=True)[0]

    return result['momentos'], result['successPercentage']


if __name__ == '__main__':
    testing()
