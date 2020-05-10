
import cv2
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

from apps.feature_extraction import FeatureComputator

def reconocedor(X):
    # Recibe una foto de pared o no pared
    # inputs
    # X ~ es un numpy.array de 64x64x3

    # output
    # resultado ~ 1 o 2 dependiendo de si está rayada o no lo está
           # 1. rayada
           # 2. no rayada

    # Ejemplo: reconecedor(A) = 1 , donde A representa una foto de una pared rayada

    # Obtenemos los datos del filtrado de features obtenido en main.py
    with open('data/reconocedor.json', 'r') as file:
        data = json.loads(file.read())
        s_clean = np.array(data['s_clean'])
        a = np.array(data['a'])
        b = np.array(data['b'])
        s_sfs = np.array(data['s_sfs'])

    # Obtenemos los datos de training
    with open('data/paredes_data.json', 'r') as file:
        training_data = json.loads(file.read())

    # Datos de training
    X_train, labels_train = training_data['feature_values_train'], training_data['labels_train']
    X_train, labels_train = np.array(X_train), np.array(labels_train)
    X_train = X_train[:, s_clean]        # Paso 3: clean
    X_train = X_train*a + b              # Paso 4: normalizacion
    X_train = X_train[:, s_sfs]          # Paso 5: SFS

    # Muestra
    features = np.array(FeatureComputator(X))
    features = features[s_clean]        # Paso 3: clean
    features = features*a + b           # Paso 4: normalizacion
    features = features[s_sfs]          # Paso 5: SFS
    features = features.reshape(1, -1)  # Reshape para el clasificador

    # Clasificamos la muestra
    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)
    Y_pred = knn.predict(features)

    return Y_pred[0]


def reconocedorTest():
    rayada = cv2.imread('img/testing/rayada/P1_04131.png')
    print('Reconocedor(rayada) =', reconocedor(rayada))

    no_rayada = cv2.imread('img/testing/no_rayada/P0_04041.png')
    print('Reconocedor(no_rayada) =', reconocedor(no_rayada))


if __name__ == '__main__':
    reconocedorTest()
