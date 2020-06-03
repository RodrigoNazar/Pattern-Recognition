
import numpy as np
from pybalu.feature_selection import clean, sfs
from pybalu.feature_transformation import normalize
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy import stats
import json
import os

from apps.classifier_performance import performance, confusionMatrix

# Útiles

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

def get_class_by_name(name):
    return int(name.split('_')[0][-1])

# Estrategias

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

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS

    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)

    correct = 0
    total = X_test.shape[0]/10

    for sample in groups['test']:
        patch_data = np.array([])
        for patch in groups['test'][sample]:

            features = X_test[groups['test'][sample][patch], :].reshape(1, -1)
            patch_data = np.append(patch_data, knn.predict(features)[0])

        if get_class_by_name(sample) == stats.mode(patch_data)[0][0]:
            correct += 1


    return correct*100/total


def strategy02(X_train, labels_train, X_test, labels_test, groups):

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

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS

    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)

    correct = 0
    total = X_test.shape[0]/10

    for sample in groups['test']:
        patch_data = np.array([])
        for patch in groups['test'][sample]:

            features = X_test[groups['test'][sample][patch], :].reshape(1, -1)
            patch_data = np.append(patch_data, knn.predict(features)[0])

        if get_class_by_name(sample) == stats.mode(patch_data)[0][0]:
            correct += 1


    return correct*100/total


def strategy03(X_train, labels_train, X_test, labels_test, groups):

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

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS

    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)

    correct = 0
    total = X_test.shape[0]/10

    for sample in groups['test']:
        patch_data = np.array([])
        for patch in groups['test'][sample]:

            features = X_test[groups['test'][sample][patch], :].reshape(1, -1)
            patch_data = np.append(patch_data, knn.predict(features)[0])

        if get_class_by_name(sample) == stats.mode(patch_data)[0][0]:
            correct += 1


    return correct*100/total


def strategy04(X_train, labels_train, X_test, labels_test, groups):

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

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS

    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)

    correct = 0
    total = X_test.shape[0]/10

    for sample in groups['test']:
        patch_data = np.array([])
        for patch in groups['test'][sample]:

            features = X_test[groups['test'][sample][patch], :].reshape(1, -1)
            patch_data = np.append(patch_data, knn.predict(features)[0])

        if get_class_by_name(sample) == stats.mode(patch_data)[0][0]:
            correct += 1


    return correct*100/total


def strategy05(X_train, labels_train, X_test, labels_test, groups):

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

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS

    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)

    correct = 0
    total = X_test.shape[0]/10

    for sample in groups['test']:
        patch_data = np.array([])
        for patch in groups['test'][sample]:

            features = X_test[groups['test'][sample][patch], :].reshape(1, -1)
            patch_data = np.append(patch_data, knn.predict(features)[0])

        if get_class_by_name(sample) == stats.mode(patch_data)[0][0]:
            correct += 1


    return correct*100/total
