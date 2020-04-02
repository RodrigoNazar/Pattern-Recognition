
from apps.feature_setup import setupImgsFeatures
from apps.testing import testImages


def main():

    # Obtenemos las características de las imágenes otorgadas
    # setupImgsFeatures('img/Training_01.png', 'img/Training_02.png',
    #                   'img/Testing.png')

    # Realizamos el testing de la imagen de testing
    testImages()

if __name__ == '__main__':
    main()
