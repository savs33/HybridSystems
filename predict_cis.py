import io_data
import numpy as np
import pandas as pd

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_squared_error


def get_antecedents_consequent():

    trans = ctrl.Antecedent(np.arange(0, 9000, 500), 'trans')
    bal = ctrl.Antecedent(np.arange(0, 40000, 500), 'bal')

    age = ctrl.Antecedent(np.arange(16, 90, 5), 'age')
    gender = ctrl.Antecedent(np.arange(0, 2, 1), 'gender')
    education = ctrl.Antecedent(np.arange(0, 2, 1), 'education')
    income = ctrl.Antecedent(np.arange(0, 20000, 500), 'income')
    occupation = ctrl.Antecedent(np.arange(0, 3, 1), 'occupation')

    cis = ctrl.Consequent(np.arange(0, 10, 1), 'cis')
    return (trans, bal, age, gender, education, income, occupation, cis)


def get_memb_fns():

    (trans, bal, age, gender, education, income,
     occupation, cis) = get_antecedents_consequent()

    trans['low'] = fuzz.trapmf(trans.universe, [0, 1000, 1580, 1738])
    trans['avg'] = fuzz.trapmf(trans.universe, [1580, 1738, 2489, 2737.9])
    trans['high'] = fuzz.trapmf(trans.universe, [2489, 2737.9, 9000, 9900])

    bal['avg'] = fuzz.trapmf(bal.universe, [20069, 22075.9, 29403, 29700])
    bal['high'] = fuzz.trapmf(bal.universe, [29403, 29700, 80810, 88000])

    age['young'] = fuzz.trapmf(age.universe, [16, 22, 34, 40])
    age['middle'] = fuzz.trapmf(age.universe, [34, 40, 50, 55])
    age['old'] = fuzz.trapmf(age.universe, [50, 55, 90, 90])

    gender['male'] = fuzz.trimf(gender.universe, [0, 0, 1])
    gender['female'] = fuzz.trimf(gender.universe, [1, 1, 2])

    education['low'] = fuzz.trimf(education.universe, [1, 1, 2])
    education['high'] = fuzz.trimf(education.universe, [0, 0, 1])

    income['low'] = fuzz.trapmf(income.universe, [0, 2674, 3620.5, 4657])
    income['avg'] = fuzz.trapmf(income.universe, [3620.5, 4657, 6254, 7218])
    income['high'] = fuzz.trapmf(income.universe, [6254, 7218, 20000, 20000])

    occupation['professional'] = fuzz.trimf(occupation.universe, [0, 0, 1])
    occupation['non-professional'] = fuzz.trimf(occupation.universe, [1, 1, 2])
    occupation['retired'] = fuzz.trimf(occupation.universe, [2, 2, 3])

    cis['low'] = fuzz.trimf(cis.universe, [0, 2, 3])
    cis['mid'] = fuzz.trimf(cis.universe, [3, 4, 6])
    cis['high'] = fuzz.trimf(cis.universe, [5, 7, 10])

    return (trans, bal, age, gender, education, income,
            occupation, cis)


def get_fuzzy_ctrl_system():

    memb_fns = get_memb_fns()

    # for item in memb_fns:
    # item.view()

    (trans, bal, age, gender, education, income,
     occupation, cis) = memb_fns

    rule0 = ctrl.Rule(
        antecedent=(bal['high'] & trans['high']), consequent=cis['high'])
    rule1 = ctrl.Rule(
        antecedent=(bal['avg'] & trans['low']), consequent=cis['low'])
    rule2 = ctrl.Rule(
        antecedent=(occupation['retired']), consequent=cis['low'])
    rule3 = ctrl.Rule(
        antecedent=(age['old']), consequent=cis['low'])
    rule4 = ctrl.Rule(
        antecedent=(income['avg'] | income['high']), consequent=cis['mid'])
    rule5 = ctrl.Rule(
        antecedent=(income['low']), consequent=cis['low'])
    system = ctrl.ControlSystem(
        rules=[rule0, rule1, rule2, rule3, rule4, rule5])
    ctrl_system = ctrl.ControlSystemSimulation(system)
    return ctrl_system


def get_single_predn(ctrl_system, inputs):

    trans, bal, age, income, occupation = inputs
    occupation_dict = {'finance': 0, 'IT': 0, 'medicine': 0, 'legal': 0,
                       'government': 1, 'manuf': 1, 'education': 1, 'construct': 1,
                       'retired': 2}
    occupation = occupation_dict[occupation]
    ctrl_system.input['trans'] = trans
    ctrl_system.input['bal'] = bal
    ctrl_system.input['age'] = age
    ctrl_system.input['income'] = income
    ctrl_system.input['occupation'] = occupation
    ctrl_system.compute()
    cis = ctrl_system.output['cis']
    return cis


def predict_cis(X):

    req_X = X[[' avtrans', ' avbal', ' age',  ' income', ' occupation']]
    req_X = req_X.values

    ctrl_system = get_fuzzy_ctrl_system()
    cis = np.array([get_single_predn(ctrl_system, inp) for inp in req_X])
    return cis


def evaluate_model(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    print('MSE : ', mse)


if __name__ == '__main__':
    X_test, y_test = io_data.load_orig_dataset('test')

    cis_pred = predict_cis(X_test)

    y_true = y_test[['cust Investment Potential Score ']].values.flatten()
    y_pred = cis_pred.flatten()
    evaluate_model(y_true, y_pred)

    df = pd.DataFrame()
    df['index'] = np.arange(1001, 5001)
    # df['cis'] = pd.Series(y_true)
    df['cis'] = pd.Series(cis_pred)
    df.to_csv('data/pred_cis.csv', index=False)

    # X_train, y_train = io_data.load_orig_dataset('train')
    # X_test, y_test = io_data.load_orig_dataset('test')

    # cis_pred_train = predict_cis(X_train)
    # cis_pred_test = predict_cis(X_test)

    # df = pd.DataFrame()
    # df['cis'] = pd.Series(cis_pred_train)
    # df.to_csv('tmp/cis_pred_train.csv', index=False)

    # df = pd.DataFrame()
    # df['cis'] = pd.Series(cis_pred_test)
    # df.to_csv('tmp/cis_pred_test.csv', index=False)