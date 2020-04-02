
from apps.feature_setup import setupImgsFeatures


def main():

    # Obtenemos las características de las imágenes otorgadas
    setupImgsFeatures('img/Training_01.png', 'img/Training_02.png',
                      'img/Testing.png')


if __name__ == '__main__':
    main()
