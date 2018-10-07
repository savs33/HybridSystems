import io_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helpers import convt_to_one_hot


def replace_categorical(train_X, test_X):

    replace_dict = {
        ' sex': {'M': 0, 'F': 1},
        ' mstatus': {'divorced': 3, 'married': 1, 'single': 0, 'widowed': 4},
        ' occupation': {'finance': 0, 'IT': 0, 'medicine': 0, 'legal': 0,
                        'government': 1, 'manuf': 1, 'education': 1, 'construct': 1,
                        'retired': 2},
        ' education': {'postgrad': 2, 'professional': 3, 'secondary': 0, 'tertiary': 1},
    }

    train_X.replace(replace_dict, inplace=True)
    test_X.replace(replace_dict, inplace=True)
    return train_X, test_X


def replace_cat_targets(train_y, test_y):

    replace_dict = {
        'status': {'A': 0, 'B': 1, 'None': 2},
    }

    train_y.replace(replace_dict, inplace=True)
    test_y.replace(replace_dict, inplace=True)
    return train_y, test_y


def get_clf_dataset():
    train_X, train_y = io_data.load_orig_dataset('train')
    test_X, test_y = io_data.load_orig_dataset('test')

    train_X, test_X = replace_categorical(train_X, test_X)
    train_y, test_y = replace_cat_targets(train_y, test_y)

    test_y = test_y[['status']]

    return (train_X, train_y), (test_X, test_y)

if __name__ == '__main__':
    get_clf_dataset()
