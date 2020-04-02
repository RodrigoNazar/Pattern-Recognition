
import cv2
from matplotlib import pyplot as plt
from functools import reduce
import numpy as np
import argparse
import json
import os


def huDiference(hu1, hu2, moments):
    diff = [(hu1[moment], hu2[moment]) for moment in moments]

    return sum(map(lambda x: (x[0] - x[1])**2, diff))**.5


def knn(list, neighbors):
    list = sorted(list, key=lambda i: i['diff'])[:neighbors]
    rep = [letter['letter'] for letter in list]
    rep = {rep.count(i): i for i in rep}
    maximum = max([i for i in rep.keys()])
    rep = rep[maximum]

    return rep

def testImages():

    print('\n2) Realizando el test de la imagen de testing...')

    # Obtenemos las features de las imágenes de training
    training_files = [file for file in os.listdir('data/') if 'Training' in file]

    training_data = []
    for file in training_files:
        with open('data/' + file, 'r') as json_file:
            training_data.append(json.load(json_file))

    training_letters = []
    for data in training_data:
        for object in data['objects']:
            training_letters.append(object)

    # Obtenemos las features de las imágenes de testing
    testing_files = [file for file in os.listdir('data/') if 'Testing' in file]

    testing_data = []
    for file in testing_files:
        with open('data/' + file, 'r') as json_file:
            testing_data.append(json.load(json_file))

    # Momentos de hu que se consideran en el experimento
    huMomentsUsed = ['phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7']

    results = []
    # Recorremos los archivos de testing
    for file in testing_data:
        # Leemos cada letra dentro de esta imagen
        for letter in file['objects']:
            selected_letters = []
            # Hacemos la comparación con cada letra del training
            for training_letter in training_letters:
                # Si la diferencia euclidiana de los datos es menor que la
                # tolerancia, entonces lo agregamos

                difference = huDiference(letter['hu-moments'],
                               training_letter['hu-moments'],
                               huMomentsUsed)

                selected_letter = {
                    'letter': training_letter['letter'],
                    'diff': difference
                }
                selected_letters.append(selected_letter)

            # Obtenemos el resultado consultando a los 5 vecinos más cercanos
            result = knn(selected_letters, 5)

            result = {
                'test_letter': letter,
                'result': result
            }

            results.append(result)

    # Ahora, tenemos un objeto JSON con cada letra del test con su respectivo
    # representante obtenido por knn de 5 vecinos

    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    testImages()
