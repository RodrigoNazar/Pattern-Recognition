
from apps.feature_extraction import FeatureExtractor


CLASSES = ['rayada', 'no_rayada']


def main():

    feature_extractor = FeatureExtractor(classes=CLASSES)


if __name__ == '__main__':
    main()
