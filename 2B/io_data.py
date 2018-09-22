import pandas as pd
import numpy as np
import sklearn


def read_csv(dataset='train'):
    if dataset == 'valid_y':
        df = pd.read_csv('data/Cust_Actual.csv')
    elif dataset == 'valid_x':
        df = pd.read_csv('data/custdatabase.csv')
    elif dataset == 'train':
        df = pd.read_csv('data/trialPromoResults.csv')
    else:
        print('Not a valid dataset')
    return df

if __name__ == '__main__':
    df = read_csv(dataset='train')
    df = read_csv(dataset='valid_y')
    df = read_csv(dataset='valid_x')
    print(df.head())
    print(df.columns)
