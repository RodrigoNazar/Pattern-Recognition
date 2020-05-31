
import numpy as np
from pybalu.feature_selection import clean, sfs
from pybalu.feature_transformation import normalize
from sklearn.neighbors import KNeighborsClassifier as KNN
import json

from apps.classifier_performance import performance, confusionMatrix


def strategy01(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 1 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 1 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 3: Cleaning de los datos
    #   > Training: 8000 x 250
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=50, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS


    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)
    Y_pred = knn.predict(X_test)


    # *** Estadísticas y desempeño del clasificador ***
    accuracy = performance(Y_pred, labels_test)
    # print("Accuracy = " + str(accuracy))
    # confusionMatrix(Y_pred, labels_test)

    return accuracy


def strategy02(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 2 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 2 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 3: Cleaning de los datos
    #   > Training: 8000 x 250
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=10, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS


    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=5)
    knn.fit(X_train, labels_train)
    Y_pred = knn.predict(X_test)


    # *** Estadísticas y desempeño del clasificador ***
    accuracy = performance(Y_pred, labels_test)
    # print("Accuracy = " + str(accuracy))
    # confusionMatrix(Y_pred, labels_test)

    return accuracy


def strategy03(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 3 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 3 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 3: Cleaning de los datos
    #   > Training: 8000 x 250
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=3, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS


    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)
    Y_pred = knn.predict(X_test)


    # *** Estadísticas y desempeño del clasificador ***
    accuracy = performance(Y_pred, labels_test)
    # print("Accuracy = " + str(accuracy))
    # confusionMatrix(Y_pred, labels_test)

    return accuracy


def strategy04(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 4 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 4 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 3: Cleaning de los datos
    #   > Training: 8000 x 250
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=26, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS


    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=5)
    knn.fit(X_train, labels_train)
    Y_pred = knn.predict(X_test)


    # *** Estadísticas y desempeño del clasificador ***
    accuracy = performance(Y_pred, labels_test)
    # print("Accuracy = " + str(accuracy))
    # confusionMatrix(Y_pred, labels_test)

    return accuracy


def strategy05(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 5 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 5 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 3: Cleaning de los datos
    #   > Training: 8000 x 250
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=7, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS


    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=5)
    knn.fit(X_train, labels_train)
    Y_pred = knn.predict(X_test)


    # *** Estadísticas y desempeño del clasificador ***
    accuracy = performance(Y_pred, labels_test)
    # print("Accuracy = " + str(accuracy))
    # confusionMatrix(Y_pred, labels_test)

    return accuracy
