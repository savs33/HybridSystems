import pandas as pd
import numpy as np
import io_data


def convt_to_one_hot(y):

    unique_items = np.unique(y)
    num_rows = len(y)
    num_cols = len(unique_items)
    one_hot = np.zeros((num_rows, num_cols)).astype(np.bool)

    for i, item in enumerate(unique_items):
        print(i, item)
        row_idx = y == item
        one_hot[row_idx, i] = 1

    print(np.sum(one_hot, axis=0))
    return one_hot * 1


def evaluate_clf(y_true, y_pred, y_pred_prob=None):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred)
    print(cm)
    print(rep)
    print('Acc :', acc)
    print('f1 :', f1)

if __name__ == '__main__':
    test_X, test_y = io_data.load_orig_dataset('test')
    convt_to_one_hot(test_y.iloc[:, 1])
