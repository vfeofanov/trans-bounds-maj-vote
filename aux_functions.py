import numpy as np
from sklearn.datasets import load_svmlight_file


def read_dna():
    df1 = load_svmlight_file("data/dna.scale")
    x1 = df1[0].todense()
    y1 = df1[1]
    df2 = load_svmlight_file("data/dna.scale.test")
    x2 = df2[0].todense()
    y2 = df2[1]
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    # label transform to 0..K-1
    y -= 1
    return x, y


def read_pendigits():
    df = load_svmlight_file("data/pendigits")
    x = df[0].todense()
    y = df[1]
    return x, y


def partially_labeled_view(x_l, y_l, x_u, y_u):
    y_undefined = np.repeat(-1, np.shape(x_u)[0])
    y_train = np.concatenate((y_l, y_undefined))
    x_train = np.concatenate((x_l, x_u))
    y_true = np.concatenate((y_l, y_u))
    n = np.size(y_train)
    # Shuffle observations
    shuffle = np.random.choice(np.arange(n), n, replace=False)
    y_train = y_train[shuffle]
    x_train = x_train[shuffle]
    y_true = y_true[shuffle]
    y_true_unlab = y_true[y_train == -1]
    return x_train, y_train, y_true_unlab
