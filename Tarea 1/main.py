
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
        momentos = ('phi1', 'phi2', 'phi3')
        aciertos = 100
    '''

    # Obtenemos las características de las imágenes otorgadas
    setupImgsFeatures('img/Training_01.png', 'img/Training_02.png',
                      'img/Testing.png')

    # Hacemos los tests de reconocimiento en base a distintas combinaciones
    # de momentos de hu
    momentos, aciertos = huCombinationTest()
    print(FINAL.format(momentos, aciertos))

    # Obtenemos los resultados de la mejor combinación e imprimimos sus
    # estadísticas
    results = testing(momentos)
    getStatistics(results)


if __name__ == '__main__':
    main()
