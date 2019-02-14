import numpy as np
from sklearn.datasets import load_svmlight_file


class ReadDataset:
    """
    A class to read different datasets in numpy.ndarray format.
    1. DNA Data Set: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    2. Vowel Data Set: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    3. Pendigits Data Set: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    4. MNIST Data Set: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    5. SensIT Vehicle Data Set: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    """
    def __init__(self):
        self._datasets = {
            'dna': _read_dna,
            # 'mnist': _read_mnist,
            'pendigits': _read_pendigits#,
            # 'vehicle': _read_vehicle,
            # 'vowel': _read_vowel
        }

    def read(self, name):
        if name in self._datasets:
            return self._datasets[name]()
        else:
            raise KeyError("There is no dataset with this name. Check description of ReadDataset")


def _read_dna():
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


def _read_pendigits():
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
