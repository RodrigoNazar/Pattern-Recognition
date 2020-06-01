
import os
import json


def compute_groups():
    with open(os.path.join('data/', 'patches_data.json'), 'r') as file:
        data = json.loads(file.read())

    train = data['files_train']
    test = data['files_test']

    # samples = train + test

    groups = {
        'train': {},
        'test': {}
    }

    for _class in range(3):
        for num in range(1, 169):
            num = str(num).zfill(4)

            patches = {}
            for patch in range(1, 11):
                patch = str(patch).zfill(3)
                patches[patch] = None

            groups['train'][f'X0{_class}_{num}'] = patches

    for _class in range(3):
        for num in range(169, 211):
            num = str(num).zfill(4)

            patches = {}
            for patch in range(1, 11):
                patch = str(patch).zfill(3)
                patches[patch] = None

            groups['test'][f'X0{_class}_{num}'] = patches

    for indx, img in enumerate(train):
        name = img.split('/')[-1].split('.')[0]
        sample = name[:-4]
        patch = name[-3:]

        groups['train'][sample][patch] = indx
        # print(str(indx) + ' ', end='')

    for indx, img in enumerate(test):
        name = img.split('/')[-1].split('.')[0]
        sample = name[:-4]
        patch = name[-3:]

        groups['test'][sample][patch] = indx

    return groups


def strategy01(X_train, labels_train, X_test, labels_test, groups):

    print('strategy01')

def strategy02(X_train, labels_train, X_test, labels_test, groups):

    print('strategy02')

def strategy03(X_train, labels_train, X_test, labels_test, groups):

    print('strategy03')

def strategy04(X_train, labels_train, X_test, labels_test, groups):

    print('strategy04')

def strategy05(X_train, labels_train, X_test, labels_test, groups):

    print('strategy05')
