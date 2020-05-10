mod = '''
5. Selección de Características
'''

import numpy as np


def centerOfMass(data):
    N, M = data.shape
    return np.sum(data, axis = 0)/N



def jFisher(data, labels, n_features):
    '''
    Retorna los índices de las características elegidas
    '''
    N, M = data.shape

    # Centro de masa de todas las muestras
    barZ = centerOfMass(data)

    # Para calular el centro de masa por clase
    classes = list(set(i for i in labels))
    indx = {i: [] for i in classes}
    for _class in classes:
        indx[_class] = np.argwhere(labels == _class)

    # Centro de masa por clase
    barZk = {_class: centerOfMass(data[indx[_class][:, 0]]) for _class in classes}


    # Para calcular la matriz de covarianza interclase
    pk = {_class: indx[_class].shape[0]/N for _class in classes}

    # Matriz de covarianza interclase
    Cb = np.zeros((M, M))
    for _class in classes:
        elem = barZk[_class] - barZ
        Cb += pk[_class]*elem.dot(np.transpose(elem))


    # Para calcular la matriz de covarianza intraclase
    Cw = np.zeros((M, M))
    for _class in classes:
        Ck = np.zeros((M, M))
        for sample in data[indx[_class][:, 0]]: # Obtengo los elementos de la clase
            elem = sample - barZk[_class]
            Ck += elem.dot(np.transpose(elem))
        Ck /= data[indx[_class][:, 0]].shape[0] - 1

        Cw += pk[_class]*Ck

    try:
        return np.trace(np.linalg.inv(Cw).dot(Cb))

    except np.linalg.LinAlgError: # Si la matriz es no invertible
        return - np.inf


def sfs(data, labels, n_features):
    selected_features = []



    for _ in range(n_features):
        pass

    return np.array(selected_features)


if __name__ == '__main__':
    print(mod)
