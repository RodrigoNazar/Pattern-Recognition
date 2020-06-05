
import numpy as np
import json

from apps.feature_extraction import FeatureExtractor
import apps.classifier1_strategies as c1
import apps.classifier2_strategies as c2
from utils.utils import saveJson

'''
Distribución de Clases:

    class_0 -> Normal
    class_1 -> Neumonia
    class_2 -> COVID19
'''
CLASSES = ['class_0', 'class_1', 'class_2']


def main():
    '''
    Desarrollo de el flujo del programa de reconocimiento de paredes rayadas

    La metodología utilizada es la que se describe en:
    https://github.com/domingomery/patrones/blob/master/clases/Cap03_Seleccion_de_Caracteristicas/presentations/PAT03_GeneralSchema.pdf
    '''

    # Paso 1: Extracción de características
    #   > 357 características por imagen
    features = FeatureExtractor(classes=CLASSES)


    # Paso 2: Definición de datos training - testing
    #   > Training: 5040 x 357
    #   > Testing: 1260 x 357
    X_train, labels_train = features['feature_values_train'], features['labels_train']
    X_test,  labels_test  = features['feature_values_test'],  features['labels_test']

    X_train, labels_train = np.array(X_train), np.array(labels_train)
    X_test,  labels_test  = np.array(X_test),  np.array(labels_test)

    # Ejecución de las estrategias para el clasificador número 1
    classifier1 = {
        'strategy01': c1.strategy01(X_train, labels_train, X_test, labels_test),
        'strategy02': c1.strategy02(X_train, labels_train, X_test, labels_test),
        'strategy03': c1.strategy03(X_train, labels_train, X_test, labels_test),
        'strategy04': c1.strategy04(X_train, labels_train, X_test, labels_test),
        'strategy05': c1.strategy05(X_train, labels_train, X_test, labels_test),
    }
    print(json.dumps(classifier1, indent=2))

    # Ejecución de las estrategias para el clasificador número 2

    groups = c2.compute_groups() # Objeto que ordena los patches por muestra, para su procesamiento en conjunto
    classifier2 = {
        'strategy01': c2.strategy01(X_train, labels_train, X_test, labels_test, groups),
        'strategy02': c2.strategy02(X_train, labels_train, X_test, labels_test, groups),
        'strategy03': c2.strategy03(X_train, labels_train, X_test, labels_test, groups),
        'strategy04': c2.strategy04(X_train, labels_train, X_test, labels_test, groups),
        'strategy05': c2.strategy05(X_train, labels_train, X_test, labels_test, groups)
    }
    print(json.dumps(classifier2, indent=2))

    # Guardamos los resutlados
    saveJson('results',{
        'classifier1': classifier1,
        'classifier2': classifier2
    })


if __name__ == '__main__':
    main()
