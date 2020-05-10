
import numpy as np
from pybalu.feature_selection import clean, sfs
from pybalu.feature_transformation import normalize
from sklearn.neighbors import KNeighborsClassifier as KNN
from pybalu.performance_eval import performance

from apps.feature_extraction import FeatureExtractor


CLASSES = ['rayada', 'no_rayada']


def main():

    # Paso 1: Extracción de características
    #   > 4000 imágenes de training rayadas
    #   > 4000 imágenes de training no rayadas
    #   > 1000 imágenes de testing rayadas
    #   > 1000 imágenes de testing no rayadas
    #   > 357 características por imagen
    features = FeatureExtractor(classes=CLASSES)


    # Paso 2: Definición de datos training - testing
    #   > Training: 8000 x 357
    #   > Testing: 2000 x 357
    X_train, labels_train = features['feature_values_train'], features['labels_train']
    X_test,  labels_test  = features['feature_values_test'],  features['labels_test']

    X_train, labels_train = np.array(X_train), np.array(labels_train)
    X_test,  labels_test  = np.array(X_test),  np.array(labels_test)


    # *** DEFINCION DE DATOS PARA EL TRAINING ***

    # Paso 3: Cleaning de los datos
    #   > Training: 8000 x 114
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización MinMax de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=50, method="fisher", show=True)
    X_train = X_train[:, s_sfs]


    # *** DEFINCION DE DATOS PARA EL TESTING ***

    X_test = X_test[:, s_clean]        # Paso 3: clean
    X_test = X_test*a + b              # Paso 4: normalizacion
    X_test = X_test[:, s_sfs]          # Paso 5: SFS


    # *** ENTRENAMIENTO CON DATOS DE TRAINING Y PRUEBA CON DATOS DE TESTING ***

    knn = KNN(n_neighbors=3)
    knn.fit(X_train, labels_train)
    Y_pred = knn.predict(X_test)
    accuracy = performance(Y_pred, labels_test)

    print("Accuracy = " + str(accuracy))


    # print('X_train', X_train.shape, type(X_train))
    # print('labels_train', labels_train.shape, type(labels_train))
    # print('X_test', X_test.shape, type(X_test))
    # print('labels_test', labels_test.shape, type(labels_test))




if __name__ == '__main__':
    main()
