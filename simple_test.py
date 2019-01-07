from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import RandomForestClassifier
from OVA_TSVM import *
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from self_learning import *
import time
import warnings
warnings.filterwarnings("ignore")


def read_dna():
    df1 = load_svmlight_file("data/dna.scale")
    x1 = df1[0].todense()
    y1 = df1[1]
    df2 = load_svmlight_file("data/dna.scale.t")
    x2 = df2[0].todense()
    y2 = df2[1]
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    # label transform to 0..K-1
    y -= 1
    return x, y


def partially_labeled_view(x_l, y_l, x_u, y_u):
    y_undefined = np.repeat(-1, np.shape(x_u)[0])
    y_new = np.concatenate((y_l, y_undefined))
    x_train = np.concatenate((x_l, x_u))
    y_true = np.concatenate((y_l, y_u))
    n = np.size(y_new)
    # Shuffle observations
    shuffle = np.random.choice(np.arange(n), n, replace=False)
    y_new = y_new[shuffle]
    x_train = x_train[shuffle]
    y_true = y_true[shuffle]
    y_true_unlab = y_true[y_new == -1]
    return x_train, y_new, y_true_unlab


def simple_test():
    # read and split data
    x, y = read_dna()
    x_l, x_u, y_l, y_u = train_test_split(x, y, test_size=0.99, random_state=40)
    print("shape of labeled part:")
    print(x_l.shape, y_l.shape)
    print("shape of unlabeled part:")
    print(x_u.shape, y_u.shape)
    print("class distribution of labeled examples:")
    print([np.sum(y_l == i) for i in range(len(np.unique(y)))])
    print("class distribution of unlabeled examples:")
    print([np.sum(y_u == i) for i in range(len(np.unique(y)))])

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

    # multi-class self-learning algorithm
    print("msla:")
    t0 = time.time()
    model, thetas = msla(x_l, y_l, x_u)
    y_pred = model.predict(x_u)
    print("optimal theta at each step:")
    print(thetas)
    print("accuracy:", accuracy_score(y_u, y_pred))
    print("f1-score:", f1_score(y_u, y_pred, average="weighted"))
    t1 = time.time()
    print("msla is done!")
    print("time:", t1-t0, "seconds")

    # multi-class self-learning algorithm with fixed theta
    theta = 0.7
    max_iter = 10
    print("fsla with theta={}:".format(theta))
    t0 = time.time()
    model = fsla(x_l, y_l, x_u, theta, max_iter)
    y_pred = model.predict(x_u)
    print("accuracy:", accuracy_score(y_u, y_pred))
    print("f1-score:", f1_score(y_u, y_pred, average="weighted"))
    t1 = time.time()
    print("fsla is done!")
    print("time:", t1-t0, "seconds")

    # partially labeled view
    x_train, y_new, yTestShuffled = partially_labeled_view(x_l, y_l, x_u, y_u)

    # label propagation
    print("label propagation:")
    t0 = time.time()
    label_prop_model = LabelPropagation(kernel='rbf', gamma=0.01, n_jobs=-1, tol=1e-3)
    label_prop_model.fit(x_train, y_new)
    y_pred = label_prop_model.predict(x_train[y_new == -1, :])
    print("accuracy:", accuracy_score(yTestShuffled, y_pred))
    print("f1-score:", f1_score(yTestShuffled, y_pred, average="weighted"))
    t1 = time.time()
    print("label propagation is done!")
    print("time:", t1 - t0, "seconds")

    print("tsvm:")
    t0 = time.time()
    yTestShuffled, yPred = one_vs_all_tsvm(x_l, y_l, x_u, y_u, timeout=None)
    print("accuracy:", accuracy_score(yTestShuffled, yPred))
    print("f1-score:", f1_score(yTestShuffled, yPred, average="weighted"))
    t1 = time.time()
    print("tsvm is done!")
    print("time:", t1 - t0, "seconds")


if __name__ == '__main__':
    simple_test()



