
import numpy as np

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
    X_train, labels_train = features['feature_values_train'], features['labels_train']
    X_test,  labels_test  = features['feature_values_test'],  features['labels_test']

    X_train, labels_train = np.array(X_train), np.array(labels_train)
    X_test,  labels_test  = np.array(X_test),  np.array(labels_test)


    print('X_train', X_train.shape, type(X_train))
    print('labels_train', labels_train.shape, type(labels_train))
    print('X_test', X_test.shape, type(X_test))
    print('labels_test', labels_test.shape, type(labels_test))


if __name__ == '__main__':
    main()
