mod = '''
8. Medición de desempeño
'''


def performance(Y_pred, labels_test):
    return (Y_pred == labels_test).sum() / len(Y_pred)


if __name__ == '__main__':
    print(mod)
