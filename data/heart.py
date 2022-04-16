from posixpath import split
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
    if not os.path.exists('datasets/heart'):
        os.mkdir('datasets/heart')
    if not os.path.exists('datasets/heart/heart.dat'):
        urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat', 'datasets/heart/heart.dat')
    
    heart = pd.read_csv('datasets/heart/heart.dat', sep=' ', names=['feat' + str(i) for i in range(1,14)] + ['class'])

    
    heart.iloc[heart['class'] == 2] = -1

    print(heart.head(5))
    print(heart['class'].value_counts())

    data_x = np.array(heart.iloc[:, :-1])
    data_y = np.array(heart.iloc[:, -1])

    train_x, val_test_x, train_y, val_test_y = train_test_split(data_x, data_y, test_size=(val_size+test_size), random_state=random_state)

    # normalize
    scaler = Normalizer()
    train_x = scaler.fit_transform(train_x)
    val_test_x = scaler.transform(val_test_x)

    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=test_size/(val_size+test_size), random_state=random_state)

    labeled_train_x, unlabeled_train_x, labeled_train_y, unlabeled_train_y = train_test_split(train_x, train_y, test_size=unlabeled_size, random_state=random_state)

    return labeled_train_x, labeled_train_y, unlabeled_train_x, unlabeled_train_y, val_x, val_y, test_x, test_y