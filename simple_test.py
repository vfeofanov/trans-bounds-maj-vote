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
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


def simple_test():
    # read and split data
    x, y = af.read_dna()
    x_l, x_u, y_l, y_u = train_test_split(x, y, test_size=0.99, random_state=40)
    print("shape of labeled part:")
    print(x_l.shape, y_l.shape)
    print("shape of unlabeled part:")
    print(x_u.shape, y_u.shape)
    print("class distribution of labeled examples:")
    print([np.sum(y_l == i) for i in range(len(np.unique(y)))])
    print("class distribution of unlabeled examples:")
    print([np.sum(y_u == i) for i in range(len(np.unique(y)))])
    print()

    # purely supervised classification
    print("random forest:")
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1)
    model.fit(x_l, y_l)
    y_pred = model.predict(x_u)
    print("accuracy:", accuracy_score(y_u, y_pred))
    print("f1-score:", f1_score(y_u, y_pred, average="weighted"))
    t1 = time.time()
    print("random forest is done")
    print("time:", t1-t0, "seconds")
    print()

    # multi-class self-learning algorithm
    print("msla:")
    t0 = time.time()
    model, thetas = sl.msla(x_l, y_l, x_u)
    y_pred = model.predict(x_u)
    print("optimal theta at each step:")
    print(thetas)
    print("accuracy:", accuracy_score(y_u, y_pred))
    print("f1-score:", f1_score(y_u, y_pred, average="weighted"))
    t1 = time.time()
    print("msla is done!")
    print("time:", t1-t0, "seconds")
    print()

    # multi-class self-learning algorithm with fixed theta
    theta = 0.7
    max_iter = 10
    print("fsla with theta={}:".format(theta))
    t0 = time.time()
    model = sl.fsla(x_l, y_l, x_u, theta, max_iter)
    y_pred = model.predict(x_u)
    print("accuracy:", accuracy_score(y_u, y_pred))
    print("f1-score:", f1_score(y_u, y_pred, average="weighted"))
    t1 = time.time()
    print("fsla is done!")
    print("time:", t1-t0, "seconds")
    print()

    # partially labeled view
    x_train, y_train, y_u_shuffled = af.partially_labeled_view(x_l, y_l, x_u, y_u)

    # label propagation
    print("label propagation:")
    t0 = time.time()
    label_prop_model = LabelPropagation(kernel='rbf', gamma=0.01, n_jobs=-1, tol=1e-3)
    label_prop_model.fit(x_train, y_train)
    y_pred = label_prop_model.predict(x_train[y_train == -1, :])
    print("accuracy:", accuracy_score(y_u_shuffled, y_pred))
    print("f1-score:", f1_score(y_u_shuffled, y_pred, average="weighted"))
    t1 = time.time()
    print("label propagation is done!")
    print("time:", t1 - t0, "seconds")
    print()

    # tsvm
    print("tsvm:")
    t0 = time.time()
    y_u_shuffled, y_pred = tsvm.ova_tsvm(x_l, y_l, x_u, y_u, db_name="dna", timeout=None)
    print("accuracy:", accuracy_score(y_u_shuffled, y_pred))
    print("f1-score:", f1_score(y_u_shuffled, y_pred, average="weighted"))
    t1 = time.time()
    print("tsvm is done!")
    print("time:", t1 - t0, "seconds")


if __name__ == '__main__':
    simple_test()



