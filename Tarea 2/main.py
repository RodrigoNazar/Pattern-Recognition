
from apps.feature_extraction import FeatureExtractor


CLASSES = ['rayada', 'no_rayada']


def main():

    features = FeatureExtractor(classes=CLASSES)


if __name__ == '__main__':
    main()
