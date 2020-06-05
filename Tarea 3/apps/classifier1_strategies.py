
import numpy as np
from pybalu.feature_selection import clean, sfs
from pybalu.feature_transformation import normalize, pca
import json

# Clasificadores
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from apps.classifier_performance import performance, confusionMatrix


'''
Las estructuras de las estrategias empleadas se basaron en la estructura de la actividad en clases:
https://github.com/domingomery/patrones/tree/master/clases/Cap03_Seleccion_de_Caracteristicas/ejercicios/PCA_SFS
'''


# Útiles

def classifier_tests(X_train, labels_train, X_test, labels_test):
    '''
    Rutina de comparación entre distintos clasificadores.

    Código inspirado en:
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    '''

    names = [
        "Nearest Neighbors 1",
        "Nearest Neighbors 3",
        "Nearest Neighbors 5",
        "Linear SVM",
        "RBF SVM",
        "Neural Net"
    ]

    classifiers = [
        KNeighborsClassifier(1),
        KNeighborsClassifier(3),
        KNeighborsClassifier(5),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        MLPClassifier(alpha=1, max_iter=1000)
    ]

    results = {}

    # Probamos cada uno de los clasificadores
    for name, classifier in zip(names, classifiers):
        classifier.fit(X_train, labels_train)
        Y_pred = classifier.predict(X_test)
        accuracy = performance(Y_pred, labels_test)

        results[name] = accuracy

    return results

# Rutinas

def strategy01(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 1 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 1 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 1: Cleaning de los datos
    #   > Training: 5040 x 82
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]

    # Paso 2: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 3: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 5040 x 50
    s_sfs = sfs(X_train, labels_train, n_features=50, method="fisher")
    X_train = X_train[:, s_sfs]

    # Paso 4: PCA
    #         > Training: 5040 x 10
    X_train, _, A, Xm, _ = pca(X_train, n_components=10)


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 1: Clean
    X_test = X_test*a + b              # Paso 2: Normalizacion
    X_test = X_test[:, s_sfs]          # Paso 3: SFS
    X_test = np.matmul(X_test - Xm, A) # Paso 4: PCA

    return classifier_tests(X_train, labels_train, X_test, labels_test)


def strategy02(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 2 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 2 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 1: Clean
    #         > Training: 5040 x 82
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]

    # Paso 2: PCA de 70 componentes
    #         > Training: 5040 x 70
    X_train, _, A, Xm, _ = pca(X_train, n_components=70)

    # Paso 3: Normalizacion
    #         > Training: 5040 x 70
    X_train, a, b = normalize(X_train)

    # Paso 4: SFS
    #         > Training: 5040 x 20
    s_sfs = sfs(X_train, labels_train, n_features=20, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***
    X_test = X_test[:, s_clean]        # Paso 1: Clean
    X_test = np.matmul(X_test - Xm, A) # Paso 2: PCA
    X_test = X_test*a + b              # Paso 3: Normalizacion
    X_test = X_test[:, s_sfs]          # Paso 4: SFS

    return classifier_tests(X_train, labels_train, X_test, labels_test)


def strategy03(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 3 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 3 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 1: Clean
    #         > Training: 5040 x 82
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]

    # Paso 2: Normalizacion
    #         > Training: 5040 x 82
    X_train, a, b = normalize(X_train)

    # Paso 3: SFS
    #         > Training: 5040 x 80
    s_sfs = sfs(X_train,labels_train,n_features=80,method="fisher")
    X_train = X_train[:, s_sfs]


    # Paso 4: PCA
    #         > Training: 5040 x 20
    X_train, _, A, Xm, _ = pca(X_train, n_components=20)


    # Paso 5: SFS
    #	      > Training: 5040 x 15
    s_sfs2 = sfs(X_train,labels_train,n_features=15,method="fisher")
    X_train = X_train[:, s_sfs2]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 1: Clean
    X_test = X_test*a + b              # Paso 2: Normalizacion
    X_test = X_test[:, s_sfs]          # Paso 3: SFS
    X_test = np.matmul(X_test - Xm, A) # Paso 4: PCA
    X_test = X_test[:, s_sfs2]         # Paso 5: SFS

    return classifier_tests(X_train, labels_train, X_test, labels_test)


def strategy04(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 4 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 4 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 1: Cleaning de los datos
    #   > Training: 5040 x 82
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 2: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 3: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 5040 x 26
    s_sfs = sfs(X_train, labels_train, n_features=26, method="fisher")
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS

    return classifier_tests(X_train, labels_train, X_test, labels_test)


def strategy05(X_train, labels_train, X_test, labels_test):
    '''
    Estrategia número 5 para el primer clasificador
    '''

    print('\nEjecutando la estrategia número 5 del primer clasificador...')

    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 1: Clean
    #         > Training: 5040 x 82
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]

    # Paso 2: PCA
    #         > Training: 5040 x 82
    X_train, _, A1, Xm1, _ = pca(X_train, n_components=X_train.shape[1])

    # Paso 3: Normalizacion
    #         > Training: 5040 x 82
    X_train, a, b = normalize(X_train)

    # Paso 4: SFS
    #         > Training: 5040 x 80
    s_sfs = sfs(X_train,labels_train,n_features=80,method="fisher")
    X_train = X_train[:, s_sfs]
    X_train_sfs80 = X_train.copy()

    # Paso 5: PCA
    #         > Training: 5040 x 10
    X_train, _, A2, Xm2, _ = pca(X_train, n_components=10)

    #Paso 6: SFS
    #	> Trainning: 5040 x 20
    X_train = np.concatenate((X_train, X_train_sfs80), axis=1)
    s_sfs2 = sfs(X_train, labels_train, n_features=20, method="fisher")
    X_train = X_train[:, s_sfs2]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]                             # Paso 1: clean
    X_test = np.matmul(X_test - Xm1, A1)                    # Paso 2: PCA
    X_test = X_test*a + b                                   # Paso 3: normalizacion
    X_test = X_test[:, s_sfs]                               # Paso 4: SFS
    X_test_sfs80 = X_test.copy()
    X_test = np.matmul(X_test - Xm2, A2)                    # Paso 5: PCA
    X_test = np.concatenate((X_test, X_test_sfs80), axis=1)
    X_test = X_test[:, s_sfs2]                              # Paso 6: SFS

    return classifier_tests(X_train, labels_train, X_test, labels_test)
