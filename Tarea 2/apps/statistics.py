import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def huHistograms(huMomentsUsed):
    with open('data/training_data.json', 'r') as json_file:
        training_data = json.load(json_file)

    moments = {mom: {
        'A': [],
        'S': [],
        'D': [],
        'F': [],
        'G': []
    } for mom in huMomentsUsed}

    for moment in huMomentsUsed:
        for letter in ['A', 'S', 'D', 'F', 'G']:
            for elem in training_data[letter]:
                moments[moment][letter].append(elem['data'][moment])


    dataFrames = {mom: pd.DataFrame(moments[mom]) for mom in huMomentsUsed}

    # Aplicamos una transformación a escala logaritmica
    log = lambda x: -np.log10(x)
    for dataframe in dataFrames:
        dataFrames[dataframe] = dataFrames[dataframe].apply(log)

    for moment in huMomentsUsed:
        titles = {
            'phi1': r'$\phi 1$',
            'phi2': r'$\phi 2$',
            'phi3': r'$\phi 3$',
            'phi4': r'$\phi 4$',
            'phi5': r'$\phi 5$',
            'phi6': r'$\phi 6$',
            'phi7': r'$\phi 7$',
        }
        dataFrames[moment].plot.kde(title=titles[moment])


def huScatter(huMomentsUsed):

    if len(huMomentsUsed) != 3:
        raise Exception

    with open('data/training_data.json', 'r') as json_file:
        training_data = json.load(json_file)

    moments = {mom: {
        'A': [],
        'S': [],
        'D': [],
        'F': [],
        'G': []
    } for mom in huMomentsUsed}

    for moment in huMomentsUsed:
        for letter in ['A', 'S', 'D', 'F', 'G']:
            for elem in training_data[letter]:
                moments[moment][letter].append(elem['data'][moment])


    dataFrames = {mom: pd.DataFrame(moments[mom]) for mom in huMomentsUsed}

    # Aplicamos una transformación a escala logaritmica
    log = lambda x: -np.log10(x)
    for dataframe in dataFrames:
        dataFrames[dataframe] = dataFrames[dataframe].apply(log)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*',
                      'h', 'H', 'D', 'd', 'P', 'X')

    for inx, letter in enumerate(('A', 'S', 'D', 'F', 'G')):
        fields = []

        for moment in huMomentsUsed:
            fields.append(dataFrames[moment][letter])

        xs, ys, zs = fields
        scatter = ax.scatter(xs, ys, zs, marker=filled_markers[inx])

    titles = {
        'phi1': r'$\phi 1$',
        'phi2': r'$\phi 2$',
        'phi3': r'$\phi 3$',
        'phi4': r'$\phi 4$',
        'phi5': r'$\phi 5$',
        'phi6': r'$\phi 6$',
        'phi7': r'$\phi 7$',
    }

    ax.set_xlabel(titles[huMomentsUsed[0]])
    ax.set_ylabel(titles[huMomentsUsed[1]])
    ax.set_zlabel(titles[huMomentsUsed[2]])
    plt.legend(('A', 'S', 'D', 'F', 'G'))


def successPercentage(results):

    correct = 0
    total = len(results)

    for result in results:
        if result['answer'] == result['output']:
            correct += 1

    return correct/total*100


def getStatistics(results):
    print('\n3) Obteniendo las estadísticas de la prueba...')

    print(f'\tEl porcentaje de aciertos es: {successPercentage(results)}%')

    huHistograms(['phi1', 'phi2', 'phi3'])

    huScatter(['phi1', 'phi2', 'phi3'])

    plt.show()


if __name__ == '__main__':
    print('Statistics module')
    huHistograms(['phi1', 'phi2', 'phi3'])
    huScatter(['phi1', 'phi2', 'phi3'])
    plt.show()
