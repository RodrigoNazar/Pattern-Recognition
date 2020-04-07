
from apps.feature_setup import setupImgsFeatures
from apps.testing import testing
from apps.statistics import getStatistics


def main():

    # Obtenemos las características de las imágenes otorgadas
    setupImgsFeatures('img/Training_01.png', 'img/Training_02.png',
                      'img/Testing.png')

    # Realizamos el testing de la imagen de testing
    # results = testing()

    # Obtenemos las estadísticas del método
    # getStatistics(results)



if __name__ == '__main__':
    main()
