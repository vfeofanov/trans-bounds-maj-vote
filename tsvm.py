import numpy as np
import pandas as pd
import subprocess
import os
from sklearn.datasets import dump_svmlight_file
import aux_functions as af


def create_folders(db_name, num_exp):
    try:
        os.mkdir("output")
    except FileExistsError:
        pass
    try:
        os.mkdir("output/"+db_name)
    except FileExistsError:
        pass
    try:
        os.mkdir("output/"+db_name+"/tsvm")
    except FileExistsError:
        pass
    if num_exp is not None:
        subpath = db_name+"/tsvm/"+str(num_exp)
        try:
            os.mkdir("output/" + subpath)
        except FileExistsError:
            pass
    else:
        subpath = db_name + "/tsvm"
    try:
        os.mkdir("output/"+subpath+"/files")
    except FileExistsError:
        pass
    try:
        os.mkdir("output/"+subpath+"/models")
    except FileExistsError:
        pass
    try:
        os.mkdir("output/"+subpath+"/predictions")
    except FileExistsError:
        pass


def get_pred_values(path):
    y = pd.read_csv(path, sep=' ', header=None)
    y = y[0]
    y = y.apply(lambda x: int(str.split(x, ':')[1]), 1)
    return y


def ova_tsvm(x_l, y_l, x_u, y_u, db_name="tmp", num_exp=None, timeout=None):
    K = len(np.unique(y_l))
    x_train, y_train, y_u_shuffled = af.partially_labeled_view(x_l, y_l, x_u, y_u)
    # create folders for storing results
    create_folders(db_name, num_exp)
    if num_exp is not None:
        subpath = db_name + "/tsvm/" + str(num_exp)
    else:
        subpath = db_name + "/tsvm"
    ovapreds = []
    for k in range(K):
        y_train_k = np.array(list(map(lambda label: 1 if label == k else (0 if label == -1 else -1), y_train)))
        path_file = "output/" + subpath + "/files/df_class_" + str(k)
        path_model = "output/" + subpath + "/models/model_class_" + str(k)
        path_prediction = "output/" + subpath + "/predictions/pred_class_" + str(k)
        open(path_file, 'a').close()
        dump_svmlight_file(x_train, y_train_k, path_file, zero_based=False)
        # form a run command to create process of learning tsvm
        cmd = form_cmd(path_file, path_model, path_prediction)
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, timeout=timeout)
            ovapreds.append(get_pred_values(path_prediction))
        except subprocess.TimeoutExpired:
            raise TimeoutError("The algorithm has not converged!")
    ovapreds = np.array(ovapreds).T
    y_pred = np.apply_along_axis(ova_voting, 1, ovapreds)
    return y_u_shuffled, y_pred


def ova_voting(preds_for_x):
    inds = [idx for idx in range(len(preds_for_x)) if preds_for_x[idx] == 1]
    if inds == []:
        return np.random.choice(np.arange(len(preds_for_x)), 1)[0]
    else:
        return np.random.choice(inds, 1)[0]


def form_cmd(path_file, path_model, path_prediction):
    cmd = list()
    # binary for learning
    cmd.append('./svm_light/svm_learn')
    # these options can be tried
    # cmd.append('-n')
    # cmd.append('5')
    # cmd.append('-e')
    # cmd.append('0.01')
    # classify unlabelled examples to the following file
    cmd.append('-l')
    cmd.append(path_prediction)
    # data path
    cmd.append(path_file)
    # a file for the learning model
    cmd.append(path_model)
    return cmd
