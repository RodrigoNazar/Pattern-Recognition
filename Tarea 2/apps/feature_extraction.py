'''
1. Extracción de Características
'''

classes = ['rayada', 'no_rayada']

class FeatureExtractor:
    def __init__(self, training_path='img/training',
                       testing_path='img/testing',
                       classes=classes):
        print(training_path, testing_path, classes)
