
import numpy as np
import json

from apps.feature_extraction import FeatureExtractor
import apps.classifier1_strategies as c1
import apps.classifier2_strategies as c2

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


    '''
    classifier1 = {
      "strategy01": {
        "Nearest Neighbors 1": 0.8634920634920635,
        "Nearest Neighbors 3": 0.873015873015873,
        "Nearest Neighbors 5": 0.8793650793650793,
        "Linear SVM": 0.9317460317460318,
        "RBF SVM": 0.3380952380952381,
        "Neural Net": 0.9531746031746032
      },
      "strategy02": {
        "Nearest Neighbors 1": 0.8142857142857143,
        "Nearest Neighbors 3": 0.8087301587301587,
        "Nearest Neighbors 5": 0.8119047619047619,
        "Linear SVM": 0.9126984126984127,
        "RBF SVM": 0.5404761904761904,
        "Neural Net": 0.9214285714285714
      },
      "strategy03": {
        "Nearest Neighbors 1": 0.8611111111111112,
        "Nearest Neighbors 3": 0.8706349206349207,
        "Nearest Neighbors 5": 0.8785714285714286,
        "Linear SVM": 0.9198412698412698,
        "RBF SVM": 0.3484126984126984,
        "Neural Net": 0.930952380952381
      },
      "strategy04": {
        "Nearest Neighbors 1": 0.8674603174603175,
        "Nearest Neighbors 3": 0.8833333333333333,
        "Nearest Neighbors 5": 0.9,
        "Linear SVM": 0.9238095238095239,
        "RBF SVM": 0.34365079365079365,
        "Neural Net": 0.9357142857142857
      },
      "strategy05": {
        "Nearest Neighbors 1": 0.8142857142857143,
        "Nearest Neighbors 3": 0.8087301587301587,
        "Nearest Neighbors 5": 0.8119047619047619,
        "Linear SVM": 0.9126984126984127,
        "RBF SVM": 0.5404761904761904,
        "Neural Net": 0.926984126984127
      }
    }

    classifier2 = {
      "strategy01": {
        "Nearest Neighbors 1": 92.06349206349206,
        "Nearest Neighbors 3": 92.06349206349206,
        "Nearest Neighbors 5": 93.65079365079364,
        "Linear SVM": 96.03174603174604,
        "RBF SVM": 34.12698412698413,
        "Neural Net": 97.61904761904762
      },
      "strategy02": {
        "Nearest Neighbors 1": 87.3015873015873,
        "Nearest Neighbors 3": 86.5079365079365,
        "Nearest Neighbors 5": 86.5079365079365,
        "Linear SVM": 91.26984126984127,
        "RBF SVM": 58.73015873015873,
        "Neural Net": 94.44444444444444
      },
      "strategy03": {
        "Nearest Neighbors 1": 89.68253968253968,
        "Nearest Neighbors 3": 93.65079365079364,
        "Nearest Neighbors 5": 90.47619047619048,
        "Linear SVM": 92.85714285714286,
        "RBF SVM": 34.12698412698413,
        "Neural Net": 97.61904761904762
      },
      "strategy04": {
        "Nearest Neighbors 1": 93.65079365079364,
        "Nearest Neighbors 3": 94.44444444444444,
        "Nearest Neighbors 5": 96.82539682539682,
        "Linear SVM": 96.82539682539682,
        "RBF SVM": 34.12698412698413,
        "Neural Net": 96.82539682539682
      },
      "strategy05": {
        "Nearest Neighbors 1": 87.3015873015873,
        "Nearest Neighbors 3": 86.5079365079365,
        "Nearest Neighbors 5": 86.5079365079365,
        "Linear SVM": 91.26984126984127,
        "RBF SVM": 58.73015873015873,
        "Neural Net": 94.44444444444444
      }
    }

    '''


if __name__ == '__main__':
    main()
