
import json
import os

def jsontests():
    with open(os.path.join('data/', 'patches_data.json'), 'r') as file:
        data = json.loads(file.read())

    for i in data.keys():
        print(i, len(data[i]))
