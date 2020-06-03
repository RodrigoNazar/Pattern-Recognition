
import numpy as np
from pybalu.feature_selection import clean, sfs
from pybalu.feature_transformation import normalize
from sklearn.neighbors import KNeighborsClassifier as KNN
import json
import os

from apps.classifier_performance import performance, confusionMatrix


def compute_groups():
    with open(os.path.join('data/', 'patches_data.json'), 'r') as file:
        data = json.loads(file.read())

    train = data['files_train']
    test = data['files_test']

    groups = {
        'train': {},
        'test': {}
    }

    for _class in range(3):
        for num in range(1, 169):
            num = str(num).zfill(4)

            patches = {}
            for patch in range(1, 11):
                patch = str(patch).zfill(3)
                patches[patch] = None

            groups['train'][f'X0{_class}_{num}'] = patches

    for _class in range(3):
        for num in range(169, 211):
            num = str(num).zfill(4)

            patches = {}
            for patch in range(1, 11):
                patch = str(patch).zfill(3)
                patches[patch] = None

            groups['test'][f'X0{_class}_{num}'] = patches

    for indx, img in enumerate(train):
        name = img.split('/')[-1].split('.')[0]
        sample = name[:-4]
        patch = name[-3:]

        groups['train'][sample][patch] = indx

    for indx, img in enumerate(test):
        name = img.split('/')[-1].split('.')[0]
        sample = name[:-4]
        patch = name[-3:]

        groups['test'][sample][patch] = indx

    return groups


def strategy01(X_train, labels_train, X_test, labels_test, groups):

    # Paso 3: Cleaning de los datos
    #   > Training: 8000 x 250
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=12, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    print(X_test.shape)

    X_test = X_test[:, s_clean]        # Paso 3: clean
    print(X_test.shape)
    X_test = X_test*a + b              # Paso 4: normalizacion
    print(X_test.shape)
    X_test = X_test[:, s_sfs]          # Paso 5: SFS
    print(X_test.shape)

    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)

    for sample in groups['test']:
        for patch in groups['test'][sample]:

            features = X_test[groups['test'][sample][patch], :].reshape(1, -1)
            groups['test'][sample][patch] = knn.predict(features)[0]

            print(sample,patch,groups['test'][sample][patch], '\n')

    # print(json.dumps(groups, indent=2))



    print('strategy01')

def strategy02(X_train, labels_train, X_test, labels_test, groups):

    print('strategy02')

def strategy03(X_train, labels_train, X_test, labels_test, groups):

    print('strategy03')

def strategy04(X_train, labels_train, X_test, labels_test, groups):

    print('strategy04')

def strategy05(X_train, labels_train, X_test, labels_test, groups):

    print('strategy05')
