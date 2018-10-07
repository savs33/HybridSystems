import numpy as np
import pandas as pd

import io_data


def calc_prof(customer_data):

    prod = customer_data[:, 1]
    cis = customer_data[:, 2]
    profit = np.zeros(shape=(len(customer_data), 1))
    profit[prod == 'A'] = 0.6
    profit[prod == 'B'] = 1
    profit = profit * np.expand_dims(cis, 1)
    profit = profit.flatten()
    return profit


def select_top_n_customers(n=400):
    df = pd.read_csv('data/pred_prod.csv')
    customer_data = df.values

    df = pd.read_csv('data/pred_cis.csv')
    cis = df.values[:, 1].reshape(-1, 1)
    customer_data = np.hstack([customer_data, cis])
    print(customer_data.shape)

    profit = calc_prof(customer_data)
    idx = np.argsort(profit)[-n:]
    return idx


def compare_profits(selected_cust_idx,n=400):

    X_test, y_test = io_data.load_orig_dataset('test')
    customer_data = y_test.values

    profit = calc_prof(customer_data)
    
    idx = np.argsort(profit)[-n:]
    profit = profit[idx]
    print('Actual Profit : ', np.sum(profit))

    profit = calc_prof(customer_data)
    profit = profit[selected_cust_idx]
    print('Selected Customers Profit : ', np.sum(profit))

if __name__ == '__main__':
    selected_cust_idx = select_top_n_customers(n=400)
    compare_profits(selected_cust_idx,n=400)
