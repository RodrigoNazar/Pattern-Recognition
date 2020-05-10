
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

    X_train = np.array(X_train)


    for i in X_train:
        if len(i) != 357:
            print(type(i), len(i), i[0])

    # print('X_train', X_train.shape, type(X_train))
    # print('labels_train', len(labels_train), type(labels_train))
    # print('X_test', len(X_test), type(X_test))
    # print('labels_test', len(labels_test), type(labels_test))


if __name__ == '__main__':
    main()
