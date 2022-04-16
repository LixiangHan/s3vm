import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import os
import urllib.request
import subprocess

def load_data(test_size=0.2, val_size=0.1, unlabeled_size=0.5, random_state=33):
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    if not os.path.exists('datasets/musk'):
        os.mkdir('datasets/musk')
    if not os.path.exists('datasets/musk/clean1.data'):
        urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean1.data.Z', 'datasets/musk/clean1.data.Z')
        decompress_sp('datasets/musk/clean1.data.Z', 'datasets/musk/clean1.data')
    musk = pd.read_csv('datasets/musk/clean1.data', names=['molecule_name', 'conformation_name'] + ['feat' + str(i) for i in range(1,163)] + ['class'])
    
    musk.iloc[musk['class'] == 0] = -1

    print(musk.head(5))
    print(musk['class'].value_counts())

    data_x = np.array(musk.iloc[:, 2:-1])
    data_y = np.array(musk.iloc[:, -1])
    train_x, val_test_x, train_y, val_test_y = train_test_split(data_x, data_y, test_size=(val_size+test_size), random_state=random_state)

    # normalize
    scaler = Normalizer()
    train_x = scaler.fit_transform(train_x)
    val_test_x = scaler.transform(val_test_x)

    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=test_size/(val_size+test_size), random_state=random_state)

    labeled_train_x, unlabeled_train_x, labeled_train_y, unlabeled_train_y = train_test_split(train_x, train_y, test_size=unlabeled_size, random_state=random_state)

    return labeled_train_x, labeled_train_y, unlabeled_train_x, unlabeled_train_y, val_x, val_y, test_x, test_y


def decompress_sp(src, dst):
    """Decompress function using gzip CLI program."""
    gzip_args = ' '.join(['gzip', '-d', '-c', src])
    with open(dst, 'wb') as df:
        status = subprocess.call(gzip_args, stdout=df, stderr=subprocess.DEVNULL)
    if status:
        raise(OSError('Not a .gz file.'))