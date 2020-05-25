mod = '''
8. Medición de desempeño
'''

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def confusionMatrix(Y_pred, labels_test):
    '''
    Función basada en la solución mostrada en:
    https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    '''
    df_cm = pd.DataFrame(confusion_matrix(Y_pred, labels_test), ['rayada', 'no rayada'], ['rayada', 'no rayada'])
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 20}, cmap='YlGnBu', fmt='g')
    plt.xlabel('Valores predecidos')
    plt.ylabel('Valores reales')
    plt.show()


def performance(Y_pred, labels_test):
    return (Y_pred == labels_test).sum() / len(Y_pred)


if __name__ == '__main__':
    print(mod)
