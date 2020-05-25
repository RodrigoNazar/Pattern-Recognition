
import numpy as np
from pybalu.feature_selection import clean
from pybalu.feature_transformation import normalize
from sklearn.neighbors import KNeighborsClassifier as KNN
import json

from apps.feature_extraction import FeatureExtractor
from apps.feature_selection import sfs
from apps.classifier_performance import performance, confusionMatrix


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

    La estructura del código se basó en el código de ejemplo de la actividad en clases:
    https://github.com/domingomery/patrones/tree/master/clases/Cap03_Seleccion_de_Caracteristicas/ejercicios/PCA_SFS
    '''

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
    #   > Training: 8000 x 250
    s_clean = clean(X_train)
    X_train = X_train[:, s_clean]


    # Paso 4: Normalización Mean-Std de los datos
    X_train, a, b = normalize(X_train)


    # Paso 5: Selección de características
    # Acá se utilizó el criterio de fisher
    #   > Training: 8000 x 50
    s_sfs = sfs(X_train, labels_train, n_features=50)
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
    print("Accuracy = " + str(accuracy))
    confusionMatrix(Y_pred, labels_test)

    printChoosenFeatures = True
    if printChoosenFeatures:
        feature_names = np.array(features['feature_names'])
        feature_names = feature_names[s_sfs]
        print('Las features seleccionadas por el sistema son:')
        for name in feature_names:
            print(name, end=' -- ')


    # *** Guardado de las variables para el reconocedor externo ***
    with open('data/reconocedor.json', 'w') as file:
        file.write(json.dumps({
            's_clean': s_clean.tolist(),
            'a': a.tolist(),
            'b': b.tolist(),
            's_sfs': s_sfs.tolist()
        }))





if __name__ == '__main__':
    main()
