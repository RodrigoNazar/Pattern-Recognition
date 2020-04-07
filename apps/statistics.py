import json
import pandas as pd


def successPercentage(results):

    correct = 0
    total = len(results)

    for result in results:
        if result['answer'] == result['output']:
            correct += 1

    return correct/total*100


def huMomentsComparison():
    print()

    with open('data/training_data.json', 'r') as json_file:
        training_data = json.load(json_file)

    columns = ['phi1', 'phi2', 'phi3', 'phi4', 'phi5', 'phi6', 'phi7']


    for letter in training_data:
        for data in training_data[letter]:
            print(json.dumps(data['data'], indent=2))

    # print(json.dumps(training_data, indent=2))


def getStatistics(results):
    print('\n3) Obteniendo las estadísticas de la prueba...')

    print(f'\tEl porcentaje de aciertos es: {successPercentage(results)}%')

    huMomentsComparison()


if __name__ == '__main__':
    print('Statistics module')
