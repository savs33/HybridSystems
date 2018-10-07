import pandas as pd


def load_orig_dataset(dataset):
    if dataset == 'train':
        df = pd.read_csv('data/orig_data/trialPromoResults.csv')
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y = y.to_frame('status')
        return X, y
    elif dataset == 'test':
        df = pd.read_csv('data/orig_data/custdatabase.csv')
        X = df.iloc[:, :-1]
        df = pd.read_csv('data/orig_data/Cust_Actual.csv')
        y = df
        return X, y
    else:
        print('dataset not available')
        return None


if __name__ == '__main__':
    X_train, y_train = load_orig_dataset('train')
    X_test,y_test = load_orig_dataset('test')

    for df in [X_train, y_train,X_test,y_test]:
    	print(df.columns)
    	print(df.shape)