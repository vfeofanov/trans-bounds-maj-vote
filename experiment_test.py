# classifiers
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import RandomForestClassifier
import tsvm
import self_learning as sl
# auxiliary functions
import aux_functions as af
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# other packages
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


def experiment_test(x, y, db_name, unlab_size, N=20):

    # rf, msla, fsla result matrices:
    # 1st component: experiment number
    # 2nd component: 0 - accuracy score, 1 - f1 score, 2 - runtime
    rf = np.zeros((N, 3))
    msla = np.zeros((N, 3))
    fsla = np.zeros((N, 3))
    ls = np.zeros((N, 3))
    ova_tsvm = np.zeros((N, 3))

    for n in range(N):

        # split on labeled and unlabeled parts
        x_l, x_u, y_l, y_u = train_test_split(x, y, test_size=unlab_size, random_state=n * 10)

        # display information about data split for the first iteration:
        if n == 0:
            print("data split for the first iteration:")
            print("shape of labeled part:")
            print(x_l.shape, y_l.shape)
            print("shape of unlabeled part:")
            print(x_u.shape, y_u.shape)
            print("class distribution of labeled examples:")
            print([np.sum(y_l == i) for i in range(len(np.unique(y)))])
            print("class distribution of unlabeled examples:")
            print([np.sum(y_u == i) for i in range(len(np.unique(y)))])
            print()

        # partially labeled view
        x_train, y_train, y_u_shuffled = af.partially_labeled_view(x_l, y_l, x_u, y_u)

        # purely supervised classification
        model = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1)
        t0 = time.time()
        model.fit(x_l, y_l)
        y_pred = model.predict(x_u)
        t1 = time.time()
        print("random forest is finished, experiment", n)
        rf[n, 0] = accuracy_score(y_u, y_pred)
        rf[n, 1] = f1_score(y_u, y_pred, average="weighted")
        rf[n, 2] = t1 - t0

        # label propagation
        t0 = time.time()
        label_prop_model = LabelPropagation(gamma=0.01, n_jobs=-1, tol=1e-3)
        label_prop_model.fit(x_train, y_train)
        y_pred = label_prop_model.predict(x_train[y_train == -1, :])
        t1 = time.time()
        print("label propagation is finished, experiment", n)
        ls[n, 0] = accuracy_score(y_u_shuffled, y_pred)
        ls[n, 1] = f1_score(y_u_shuffled, y_pred, average="weighted")
        ls[n, 2] = t1 - t0

        # tsvm
        t0 = time.time()
        y_u_shuffled, y_pred = tsvm.ova_tsvm(x_l, y_l, x_u, y_u, db_name=db_name, timeout=None)
        t1 = time.time()
        print("tsvm is finished, experiment", n)
        ova_tsvm[n, 0] = accuracy_score(y_u_shuffled, y_pred)
        ova_tsvm[n, 1] = f1_score(y_u_shuffled, y_pred, average="weighted")
        ova_tsvm[n, 2] = t1 - t0

        # multi-class self-learning algorithm with fixed theta
        theta = 0.7
        max_iter = 10
        t0 = time.time()
        model = sl.fsla(x_l, y_l, x_u, theta, max_iter)
        y_pred = model.predict(x_u)
        t1 = time.time()
        print("fsla is finished, experiment", n)
        fsla[n, 0] = accuracy_score(y_u, y_pred)
        fsla[n, 1] = f1_score(y_u, y_pred, average="weighted")
        fsla[n, 2] = t1 - t0

        # multi-class self-learning algorithm
        t0 = time.time()
        model, thetas = sl.msla(x_l, y_l, x_u)
        y_pred = model.predict(x_u)
        t1 = time.time()
        print("msla is finished, experiment", n)
        msla[n, 0] = accuracy_score(y_u, y_pred)
        msla[n, 1] = f1_score(y_u, y_pred, average="weighted")
        msla[n, 2] = t1 - t0

        print("experiment", n, "is done")

    acc = np.vstack((
        rf[:, 0],
        ls[:, 0],
        ova_tsvm[:, 0],
        fsla[:, 0],
        msla[:, 0]
    )).T
    f1 = np.vstack((
        rf[:, 1],
        ls[:, 1],
        ova_tsvm[:, 1],
        fsla[:, 1],
        msla[:, 1]
    )).T
    return np.mean(acc, axis=0), np.std(acc, axis=0), acc, np.mean(f1, axis=0), np.std(f1, axis=0), f1


if __name__ == '__main__':
    x, y = af.read_pendigits()
    experiment_test(x, y, db_name="pendigits", unlab_size=0.99)
