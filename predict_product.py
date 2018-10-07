import pandas as pd
import numpy as np
np.random.seed(42)
import lightgbm as lgbm
import catboost
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
import helpers
import prepare_data_clf


def get_data(scaler, one_hot, convt_cat):

    (train_X, train_y), (test_X, test_y) = prepare_data_clf.prep_clf_data(
        scaler, one_hot, convt_cat)
    train_X = train_X.values
    test_X = test_X.values
    train_y = train_y.values
    test_y = test_y.values

    return (train_X, train_y), (test_X, test_y)


def train_and_predict_lgbm():
    (train_X, train_y), (test_X, test_y) = get_data(
        scaler='none', one_hot=False, convt_cat=True)
    model = lgbm.LGBMClassifier(
        boosting_type='goss', max_depth=-1, n_estimators=1000, random_state=42)

    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    y_true = test_y
    helpers.evaluate_clf(y_true, y_pred)

    pred_for_train = model.predict(train_X)
    pred_for_test = model.predict(test_X)
    return pred_for_train, pred_for_test


def train_and_predict_catboost():
    (train_X, train_y), (test_X, test_y) = get_data(
        scaler='none', one_hot=False, convt_cat=False)

    train_y = train_y.flatten()
    test_y = test_y.flatten()

    cat_features = [1, 2, 4, 5, 6]
    train_pool = catboost.Pool(
        data=train_X, label=train_y, cat_features=cat_features)
    test_pool = catboost.Pool(
        data=test_X, label=test_y, cat_features=cat_features)

    model = catboost.CatBoostClassifier(
        loss_function='MultiClass',
        depth=None, random_seed=42, cat_features=[1, 2, 4, 5, 6], silent=False)
    model.fit(train_pool)
    y_pred = model.predict(test_pool)
    y_true = test_y

    print(y_pred.shape)
    print(np.unique(y_pred))
    helpers.evaluate_clf(y_true, y_pred)

    pred_for_train = model.predict(train_X)
    pred_for_test = model.predict(test_X)
    return pred_for_train, pred_for_test


def train_and_predict_mlp():
    (train_X, train_y), (test_X, test_y) = get_data(
        scaler='standard', one_hot=False, convt_cat=True)
    model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam',
                          alpha=1e-4, batch_size=100, learning_rate='adaptive', max_iter=500, random_state=42)

    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    y_true = test_y
    helpers.evaluate_clf(y_true, y_pred)

    pred_for_train = model.predict(train_X)
    pred_for_test = model.predict(test_X)
    return pred_for_train, pred_for_test


def train_and_predict_naive_bayes():

    (train_X, train_y), (test_X, test_y) = get_data(
        scaler='minmax', one_hot=False, convt_cat=True)
    model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    y_true = test_y
    helpers.evaluate_clf(y_true, y_pred)

    pred_for_train = model.predict(train_X)
    pred_for_test = model.predict(test_X)
    return pred_for_train, pred_for_test


def stacked_ensemble(use_saved):

    if not use_saved:

        (train_X, train_y), (test_X, test_y) = get_data(
            scaler='minmax', one_hot=False, convt_cat=False)

        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        train = train_y
        test = test_y

        train_y, test_y = train_and_predict_lgbm()
        train = np.hstack([train, train_y.reshape(-1, 1)])
        test = np.hstack([test, test_y.reshape(-1, 1)])

        train_y, test_y = train_and_predict_catboost()
        train = np.hstack([train, train_y.reshape(-1, 1)])
        test = np.hstack([test, test_y.reshape(-1, 1)])

        train_y, test_y = train_and_predict_mlp()
        train = np.hstack([train, train_y.reshape(-1, 1)])
        test = np.hstack([test, test_y.reshape(-1, 1)])

        train_y, test_y = train_and_predict_naive_bayes()
        train = np.hstack([train, train_y.reshape(-1, 1)])
        test = np.hstack([test, test_y.reshape(-1, 1)])

        print(train.shape)
        print(test.shape)

        np.save('tmp/ens_train.npy', train)
        np.save('tmp/ens_test.npy', test)

    train = np.load('tmp/ens_train.npy')
    test = np.load('tmp/ens_test.npy')

    train_X = train[:, 1:]
    train_y = train[:, 0]

    test_X = test[:, 1:]
    test_y = test[:, 0]

    model = GradientBoostingClassifier(
        n_estimators=1000, max_depth=None, random_state=42)

    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    y_true = test_y
    helpers.evaluate_clf(y_true, y_pred)

    pred_for_train = model.predict(train_X)
    return y_pred,y_true


if __name__ == '__main__':
    
    # train_and_predict_lgbm()
    # train_and_predict_catboost()
    # train_and_predict_mlp()
    # train_and_predict_naive_bayes()

    y_pred,y_true = stacked_ensemble(use_saved=True)

    pred_prod = np.array(['None']*len(y_pred))
    pred_prod[y_pred==0] = 'A'
    pred_prod[y_pred==1] = 'B'

    # import io_data
    # test_X, test_y = io_data.load_orig_dataset('test')
    # pred_prod = test_y['status'].values

    df = pd.DataFrame()
    df['index'] = np.arange(1001,5001)
    df['status'] = pred_prod
    df.to_csv('data/pred_prod.csv',index=False)
