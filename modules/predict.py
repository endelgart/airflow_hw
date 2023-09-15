# <YOUR_IMPORTS>
from typing import List, Any

import pandas as pd
import dill
import os
import json
import glob
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
#path = os.path.expanduser('C:/Users/ppmet/Air_project/airflow_hw')
path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path

def predict():

    models_list = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{models_list[-1]}', 'rb') as file:
        model = dill.load(file)

    data = pd.DataFrame(columns=['id', 'preds'])
    for file in glob.glob(f'{path}/data/test/*.json'):
        with open(file) as f:
            data1 = json.load(f)
            data3 = pd.DataFrame.from_dict([data1])
            pred = model.predict(data3)
            dict1 = {'id': data3.id, 'pred': pred}
            data2 = pd.DataFrame(dict1)
            predicted = pd.concat([data, data2], axis=0)
            data = predicted
    data.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')





if __name__ == '__main__':
    predict()
