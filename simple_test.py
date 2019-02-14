# classifiers
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import RandomForestClassifier
import tsvm
import self_learning as sl
# auxiliary functions
from aux_functions import ReadDataset, partially_labeled_view
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# other packages
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


def plot_graph(acc, f1):
    plt.subplots()
    index = np.arange(5)
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, acc, bar_width,
            alpha=opacity,
            color='b',
            label='ACC')

    plt.bar(index + bar_width, f1, bar_width,
            alpha=opacity,
            color='r',
            label='F1')

    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Performance Results')
    plt.xticks(index + bar_width, ('RF', 'LS', 'TSVM', 'FSLA', 'MSLA'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def simple_test():
    # read and split data
    read_data = ReadDataset()
    x, y = read_data.read("dna")
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

    # partially labeled view
    x_train, y_train, y_u_shuffled = partially_labeled_view(x_l, y_l, x_u, y_u)

    # purely supervised classification
    print("random forest:")
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1)
    model.fit(x_l, y_l)
    y_pred = model.predict(x_u)
    acc = [accuracy_score(y_u, y_pred)]
    f1 = [f1_score(y_u, y_pred, average="weighted")]
    print("accuracy:", acc[0])
    print("f1-score:", f1[0])
    t1 = time.time()
    print("random forest is done")
    print("time:", t1-t0, "seconds")
    print()

    # label propagation
    print("label propagation:")
    t0 = time.time()
    label_prop_model = LabelPropagation(gamma=0.01, n_jobs=-1, tol=1e-3)
    label_prop_model.fit(x_train, y_train)
    y_pred = label_prop_model.predict(x_train[y_train == -1, :])
    acc.append(accuracy_score(y_u_shuffled, y_pred))
    f1.append(f1_score(y_u_shuffled, y_pred, average="weighted"))
    print("accuracy:", acc[1])
    print("f1-score:", f1[1])
    t1 = time.time()
    print("label propagation is done!")
    print("time:", t1 - t0, "seconds")
    print()

    # tsvm
    print("tsvm:")
    t0 = time.time()
    y_u_shuffled, y_pred = tsvm.ova_tsvm(x_l, y_l, x_u, y_u, db_name="dna", timeout=None)
    acc.append(accuracy_score(y_u_shuffled, y_pred))
    f1.append(f1_score(y_u_shuffled, y_pred, average="weighted"))
    print("accuracy:", acc[2])
    print("f1-score:", f1[2])
    t1 = time.time()
    print("tsvm is done!")
    print("time:", t1 - t0, "seconds")

    # multi-class self-learning algorithm with fixed theta
    theta = 0.7
    max_iter = 10
    print("fsla with theta={}:".format(theta))
    t0 = time.time()
    model = sl.fsla(x_l, y_l, x_u, theta, max_iter)
    y_pred = model.predict(x_u)
    acc.append(accuracy_score(y_u, y_pred))
    f1.append(f1_score(y_u, y_pred, average="weighted"))
    print("accuracy:", acc[3])
    print("f1-score:", f1[3])
    t1 = time.time()
    print("fsla is done!")
    print("time:", t1-t0, "seconds")
    print()

    # multi-class self-learning algorithm
    print("msla:")
    t0 = time.time()
    model, thetas = sl.msla(x_l, y_l, x_u)
    y_pred = model.predict(x_u)
    print("optimal theta at each step:")
    print(thetas)
    acc.append(accuracy_score(y_u, y_pred))
    f1.append(f1_score(y_u, y_pred, average="weighted"))
    print("accuracy:", acc[4])
    print("f1-score:", f1[4])
    t1 = time.time()
    print("msla is done!")
    print("time:", t1-t0, "seconds")
    print()

    # plot a graph
    plot_graph(acc, f1)


if __name__ == '__main__':
    simple_test()
