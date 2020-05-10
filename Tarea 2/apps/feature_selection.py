mod = '''
5. Selección de Características
'''

import numpy as np
import json
import os
from datetime import datetime


def jFisher(data, labels):
    '''
    Retorna los índices de las características elegidas
    '''
    if len(data.shape) == 2:
        N, M = data.shape
    elif len(data.shape) == 1:
        N, M = data.shape[0], 1
    else:
        N, M = data.shape[0], data.shape[1]

    # Centro de masa de todas las muestras
    barZ = data.mean(axis=0)

    classes = list(set(i for i in labels)) # Las distintas clases

    # Indices de cada elemento de cada clase
    indx = {_class: np.argwhere(labels == _class) for _class in classes}
    # Centro de masa por clase:
    barZk = {_class: data[indx[_class][:, 0]].mean(axis=0) for _class in classes}
    # Probabilidad de ocurrencia de cada clase
    pk = {_class: indx[_class].shape[0]/N for _class in classes}

    # Matrices de covarianza inter e intra clases
    Cw, Cb = np.zeros((M, M)), np.zeros((M, M))

    for _class in classes:
        # Matriz de covarianza Interclase
        Cb_diff = (barZk[_class] - barZ).reshape((M, 1))
        Cb += pk[_class] * np.dot(Cb_diff, Cb_diff.T)

        # Matriz de covarianza Intraclase
        class_features = data[indx[_class], :]
        barZ_class = barZk[_class]

        Ck = np.zeros((M, M))
        for feature in class_features:
            Ck_diff = (feature - barZ_class).reshape((M, 1))
            Ck += np.dot(Ck_diff, Ck_diff.T)

        Ck /= class_features.shape[0] - 1

        Cw += pk[_class] * Ck


    try:
        return np.trace(np.linalg.inv(Cw).dot(Cb))

    except np.linalg.LinAlgError: # Si la matriz es no invertible
        return - np.inf


def sfs(data, labels, n_features):

    print(f'\nRealizando sfs para {n_features} características ...')

    # Si ya se calcularon las características, sólo las consultamos
    existsData = os.listdir('data/')
    if 'sfs_cache.json' in existsData:
        with open(os.path.join('data/', 'sfs_cache.json'), 'r') as file:
            file_data = json.loads(file.read())

            sameData = np.array_equal(data, np.array(file_data['data']))
            sameLabels = np.array_equal(labels, np.array(file_data['labels']))
            sameN_features = np.array_equal(n_features, np.array(file_data['n_features']))

            if sameData and sameLabels and sameN_features:
                print('Se encontraron datos de ese SFS ya calculados!')
                return file_data['selected_features']

    start = datetime.now()

    N, M = data.shape

    selected_features = []
    remaining_features = [i for i in range(M)]
    current_scores = {}

    TOLERANCE = .0001

    last_J = - np.inf

    for _ in range(n_features):
        for inx, feature in enumerate(remaining_features):
            features = selected_features + [feature]
            # print(features)
            # print(data[:, features].shape)
            current_scores[jFisher(data[:, features], labels)] = feature

        best = current_scores[max(current_scores.keys())]
        current_J = max(current_scores.keys())

        selected_features.append(best)
        remaining_features.remove(best)
        current_scores = {}

        # print(best, abs(last_J - current_J))
        print(len(selected_features), 'características escogidas...')

        if abs(last_J - current_J) < TOLERANCE:
            print('Se superó la tolerancia, delta = ', abs(last_J - current_J))
            break
        else:
            last_J = current_J



    file_data = {
        'data': data.tolist(),
        'labels': labels.tolist(),
        'n_features': n_features,
        'selected_features': selected_features
    }

    with open('data/sfs_cache.json', 'w') as file:
        file.write(json.dumps(file_data))


    print('Tiempo tomado por el SFS: ', datetime.now() - start)

    return np.array(selected_features)


if __name__ == '__main__':
    print(mod)
