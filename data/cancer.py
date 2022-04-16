import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import os
import urllib.request

def load_data(test_size=0.2, val_size=0.1, unlabeled_size=0.5, random_state=33):
    # random_state for reproducity
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    if not os.path.exists('datasets/cancer'):
        os.mkdir('datasets/cancer')
    if not os.path.exists('datasets/cancer/wdbc.data'):
        urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', 'datasets/cancer/wdbc.data')
    
    wdbc = pd.read_csv('datasets/cancer/wdbc.data', names=['ID', 'Diagnosis'] + ['feat' + str(i) for i in range(1,31)])
    
    wdbc.replace({'B': -1, 'M': 1}, inplace=True)

    print(wdbc.head(5))
    print(wdbc['Diagnosis'].value_counts())

    data_x = np.array(wdbc.iloc[:, 2:])
    data_y = np.array(wdbc.iloc[:, 1])
    train_x, val_test_x, train_y, val_test_y = train_test_split(data_x, data_y, test_size=(val_size+test_size), random_state=random_state)

    # normalize
    scaler = Normalizer()
    train_x = scaler.fit_transform(train_x)
    val_test_x = scaler.transform(val_test_x)

    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=test_size/(val_size+test_size), random_state=random_state)

    labeled_train_x, unlabeled_train_x, labeled_train_y, unlabeled_train_y = train_test_split(train_x, train_y, test_size=unlabeled_size, random_state=random_state)

    return labeled_train_x, labeled_train_y, unlabeled_train_x, unlabeled_train_y, val_x, val_y, test_x, test_y
