import numpy as np
import pandas as pd
import subprocess
from sklearn.datasets import dump_svmlight_file
from experiment_functions import Make_SSL_Train_Set, Export_SVMlight


def export_svmlight(x, y, filename="bufer"):
    open("bufer", 'a').close()
    dump_svmlight_file(x, y, ("svm_light/bufer/"+filename), zero_based=False)


def get_pred_values(path):
    y = pd.read_csv(path, sep=' ', header=None)
    y = y[0]
    y = y.apply(lambda x: int(str.split(x, ':')[1]), 1)
    return y


def one_vs_all_tsvm(xTrain, yTrain, xTest, yTest, timeout=None):
    K = len(np.unique(yTrain))
    X, y, yTestShuffled = Make_SSL_Train_Set(xTrain, yTrain, xTest, yTest, binary=False)
    ovapreds = []

    for k in range(K):
        y_k = np.array(list(map(lambda label: 1 if label == k else (0 if label == -1 else -1), y)))
        # print(np.sum(y_k==1), np.sum(y_k==0), np.sum(y_k==-1))
        Export_SVMlight(X, y_k)
        # if svmlin
        cmd = form_cmd()
        print("Starting", k)
        import time
        t0 = time.time()
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, timeout=timeout)
            ovapreds.append(get_pred_values("svm_light/bufer/predictions"))
        except subprocess.TimeoutExpired:
            u = xTest.shape[0]
            ovapreds.append(np.random.choice([-1, 1], u, replace=True))
        t1 = time.time()
        print("time:", t1-t0)
        print(k, "is done!")
    ovapreds = np.array(ovapreds).T
    print("ovapreds", ovapreds.shape)
    yPred = np.apply_along_axis(ova_voting, 1, ovapreds)
    return yTestShuffled, yPred


def ova_voting(preds_for_x):
    inds = [idx for idx in range(len(preds_for_x)) if preds_for_x[idx] == 1]
    if inds == []:
        return np.random.choice(np.arange(len(preds_for_x)), 1)[0]
    else:
        return np.random.choice(inds, 1)[0]


def form_cmd():
    cmd = list()
    # binary for learning
    cmd.append('./svm_light/svm_learn')
    # these options can be tried
    # cmd.append('-n')
    # cmd.append('3')
    # cmd.append('-e')
    # cmd.append('0.01')
    # classify unlabelled examples to the following file
    cmd.append('-l')
    cmd.append('svm_light/bufer/predictions')
    # data path
    cmd.append("svm_light/bufer/bufer")
    # a file for the learning model
    cmd.append("svm_light/bufer/model")
    return cmd
