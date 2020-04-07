import json


def successPercentage(results):

    correct = 0
    total = len(results)

    for result in results:
        if result['answer'] == result['output']:
            correct += 1

    return correct/total*100


def getStatistics(results):
    print('\n3) Obteniendo las estadísticas de la prueba...')

    print(f'\tEl porcentaje de aciertos es: {successPercentage(results)}')



if __name__ == '__main__':
    print('Statistics module')
