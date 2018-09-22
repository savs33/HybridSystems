import pandas as pd
import numpy as np
import sklearn
import io_data


def prep_categorical(df):
    categorical_columns = [' sex', ' mstatus', ' occupation', ' education']
    # enc = sklearn.preprocessing.OneHotEncoder()
    for cat_col in categorical_columns:
        dfDummies = pd.get_dummies(df[cat_col], prefix=str(cat_col)+'_cat_')
        df = pd.concat([df, dfDummies], axis=1)
    return df

if __name__ == '__main__':
    df = io_data.read_csv(dataset='train')
    df = prep_categorical(df)
    df.to_csv('data/prep_train.csv',index=False)

    df = io_data.read_csv(dataset='valid')
    df = prep_categorical(df)
    df.to_csv('data/prep_valid.csv',index=False)
