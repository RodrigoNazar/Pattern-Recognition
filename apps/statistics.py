import json
import pandas as pd
import matplotlib.pyplot as plt


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

    # print(dataFrames)
    # dataFrames['phi1'].plot.kde(title='phi1')
    dataFrames['phi2'].plot.kde(title='phi2')
    dataFrames['phi3'].plot.kde(title='phi3')
    dataFrames['phi4'].plot.kde(title='phi4')
    # dataFrames['phi5'].plot.kde(title='phi5')
    # dataFrames['phi6'].plot.kde(title='phi6')
    # dataFrames['phi7'].plot.kde(title='phi7')
    plt.show()


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

    huHistograms(['phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7'])


if __name__ == '__main__':
    print('Statistics module')
    huHistograms(['phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7'])
