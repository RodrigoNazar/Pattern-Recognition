
from apps.feature_setup import setupImgsFeatures
from apps.testing import huCombinationTest, testing
from apps.statistics import getStatistics


FINAL = '''
Finalmente, los mejores resultados fueron obtenidos usando los momentos {},
dando así un porcentaje de aciertos de {}%
'''


def main():
    '''
    SPOILER:

    Los resultados óptimos calculados por el programa son:
        momentos = ('phi2', 'phi3', 'phi4')
        aciertos = 98.88888888888889
    '''

    # Obtenemos las características de las imágenes otorgadas
    setupImgsFeatures('img/Training_01.png', 'img/Training_02.png',
                      'img/Testing.png')

    # Realizamos el testing de la imagen de testing
    momentos, aciertos = huCombinationTest()

    results = testing(momentos)

    getStatistics(results)

    print(FINAL.format(momentos, aciertos))


if __name__ == '__main__':
    main()
