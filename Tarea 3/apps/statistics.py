import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def testsPercentage(bars):

    plt.figure()

    # width of the bars
    barWidth = 0.1

    # Choose the height of the blue bars

    bars1, bars2, bars3, bars4, bars5, bars6 = bars


    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]

    # Create bars

    plt.bar(r1, bars1, width = barWidth, color = '#D97E8E', edgecolor = 'black', capsize=7, label='Nearest Neighbors 1')
    plt.bar(r2, bars2, width = barWidth, color = '#BF3056', edgecolor = 'black', capsize=7, label='Nearest Neighbors 3')
    plt.bar(r3, bars3, width = barWidth, color = '#8FA4BF', edgecolor = 'black', capsize=7, label='Nearest Neighbors 5')
    plt.bar(r4, bars4, width = barWidth, color = '#127CA6', edgecolor = 'black', capsize=7, label='Linear SVM')
    plt.bar(r5, bars5, width = barWidth, color = '#D9984A', edgecolor = 'black', capsize=7, label='RBF SVM')
    plt.bar(r6, bars6, width = barWidth, color = '#F2B279', edgecolor = 'black', capsize=7, label='Neural Net')

    # general layout
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Estrategia 1', 'Estrategia 2', 'Estrategia 3', 'Estrategia 4', 'Estrategia 5'])
    plt.ylabel('Porcentaje de Aciertos')
    plt.legend()


def getStatistics():
    print('\n3) Obteniendo las estadísticas de las pruebas...')

    with open('data/results.json', 'r') as file:
        results = json.loads(file.read())

    bars1 = [1 for _ in range(5)]
    bars2 = [2 for _ in range(5)]
    bars3 = [3 for _ in range(5)]
    bars4 = [4 for _ in range(5)]
    bars5 = [5 for _ in range(5)]
    bars6 = [6 for _ in range(5)]

    bars = (bars1, bars2, bars3, bars4, bars5, bars6)

    # Para el clasificador 1
    bars = []
    for strategy in results['classifier1']:
        percentages = []
        for classifier in results['classifier1'][strategy]:
            percentages.append(results['classifier1'][strategy][classifier])

        bars.append(percentages)

    print(list(zip(*bars)))
    print(bars)

    # testsPercentage(bars)

    # Para el clasificador 2

    # for strategy in results['classifier2']:
    #     percentages = []
    #     for classifier in results['classifier2'][strategy]:
    #         print(classifier)
    #
    # testsPercentage(bars)


    # Show graphic
    plt.show()



if __name__ == '__main__':
    print('Statistics module')
    huHistograms(['phi1', 'phi2', 'phi3'])
    huScatter(['phi1', 'phi2', 'phi3'])
    plt.show()
