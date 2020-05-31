

a = calculo_global()

def calculo_global():
    print('llamando al calculo')
    return 2

def strategy01(X_train, labels_train, X_test, labels_test):
    global a

    print('strategy01')

def strategy02(X_train, labels_train, X_test, labels_test):
    global a

    print('strategy02')

def strategy03(X_train, labels_train, X_test, labels_test):
    global a

    print('strategy03')

def strategy04(X_train, labels_train, X_test, labels_test):
    global a

    print('strategy04')

def strategy05(X_train, labels_train, X_test, labels_test):
    global a

    print('strategy05')
