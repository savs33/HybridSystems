import pandas as pd


def read_csv(dataset='train'):
    if dataset == 'valid':
        df_y = pd.read_csv('data/Cust_Actual.csv')
        df_x = pd.read_csv('data/custdatabase.csv')
        df = pd.merge(df_x, df_y, on='index', how='inner')
    elif dataset == 'train':
        df = pd.read_csv('data/trialPromoResults.csv')
    else:
        print('Not a valid dataset')
    return df

if __name__ == '__main__':
    df = read_csv(dataset='train')
    print(df.head())
    print(df.columns)
    print(df.shape)

    df = read_csv(dataset='valid')
    print(df.head())
    print(df.columns)
    print(df.shape)
