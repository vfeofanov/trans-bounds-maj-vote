import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pyximport;
pyximport.install()
import self_learning_cython as slc


def joint_bayes_risk(margin, pred, i, j, theta, samplingRate=50):
    # li = \sum_{x\in X_U} \I{y=i} =approx.= \sum_{x\in X_U} m_Q(x,i)
    li = np.sum(margin[:, i])
    margins = margin[:, j]
    # gammas = sorted(list(set(margins[margins > theta])))
    gammas = theta + (1 - theta) * (np.arange(samplingRate) + 1) / samplingRate
    infimum = 1e+05
    upperBounds = []
    # for gamma in gammas:
    for n in range(np.size(gammas)):
        gamma = gammas[n]
        I_ij = np.sum(margin[np.array((margins < gamma) & (margins >= theta)), i]) / li
        K_ij = np.dot(margin[:, i], np.array(pred == j) * margins) / li
        # M-less of gamma
        Mg_ij = np.dot(margin[:, i], np.array(margins < gamma) * margins) / li
        # M-less of theta
        Mt_ij = np.dot(margin[:, i], np.array(margins < theta) * margins) / li
        A = K_ij + Mt_ij - Mg_ij
        upperBound = I_ij + (A * (A > 0)) / gamma
        upperBounds.append(upperBound)
        if upperBound < infimum:
            infimum = upperBound
        if n > 3:
            if upperBounds[-1] > upperBounds[-2] and upperBounds[-2] >= upperBounds[-3]:
                break
    return infimum


def optimal_threshold_vector(margin, pred, K, samplingRate=50):
    theta = []

    def Reduction(matrix, margin):
        K = margin.shape[1]
        u = margin.shape[0]
        countClass = np.array([np.sum(margin[:, j]) for j in range(K)])
        return (1 / u) * np.dot(countClass, np.sum(matrix, axis=1))

    u = margin.shape[0]
    for k in range(K):
        # A set of possible thetas:
        theta_min = np.min(margin[:, k])
        theta_max = np.max(margin[:, k])
        thetas = theta_min + np.arange(samplingRate) * (theta_max - theta_min) / samplingRate
        JBR = []
        BE = []
        for n in range(samplingRate):
            matrix = np.zeros((K, K))
            for i in range(K):
                if i == k:
                    continue
                else:
                    matrix[i, k] = joint_bayes_risk(margin, pred, i, k, thetas[n])
                    if (i == 0) and (k == 1):
                        JBR.append(matrix[i, k])

            pbl = (1 / u) * np.sum((margin[:, k] >= thetas[n]) & (pred == k))
            if pbl == 0:
                pbl = 1e-15
            BE.append(Reduction(matrix, margin)/pbl)
            if n > 3:
                if BE[-1] > BE[-2] and BE[-2] >= BE[-3]:
                    break
        BE = np.array(BE)
        num = np.argmin(BE)
        if type(num) is list:
            num = num[0]
        theta.append(thetas[num])
    return np.array(theta)


def msla(x_l, y_l, x_u, cython=True, **kwargs):
    """
    A margin-based self-learning algorithm.
    :param x_l: Labeled observations.
    :param y_l: Labels.
    :param x_u:  Unlabeled data. Will be used for learning.
    :param cython:  Whether or not to use cython code, which gives speedup in computation. The default value is True.
    :return: The final classification model H that has been trained on (x_l, y_l)
    and pseudo-labeled (x_u, yPred)
    """

    if 'n_estimators' not in kwargs:
        n_est = 200
    else:
        n_est = kwargs['n_estimators']

    if 'random_state' not in kwargs:
        rand_state = None
    else:
        rand_state = kwargs['random_state']

    classifier = RandomForestClassifier(n_estimators=n_est, oob_score=True, n_jobs=-1, random_state=rand_state)
    l = x_l.shape[0]
    sample_distr = np.repeat(1 / l, l)
    K = np.unique(y_l).shape[0]
    b = True
    thetas = []
    while b:
        u = x_u.shape[0]
        # Learn a classifier
        H = classifier
        H.fit(x_l, y_l, sample_weight=sample_distr)
        margin_u = H.predict_proba(x_u)
        pred_u = np.argmax(margin_u, axis=1)

        # Find a threshold minimizing Bayes conditional error
        if cython:
            theta = slc.c_optimal_threshold_vector(margin_u, pred_u, K)
        else:
            theta = optimal_threshold_vector(margin_u, pred_u, K)
        thetas.append(theta)

        # Select observations with argmax margin more than corresponding theta
        selection = np.array(margin_u[np.arange(u), pred_u] >= theta[pred_u])
        x_s = x_u[selection, :]
        y_s = pred_u[selection]
        # Stop if there is no anything to add:
        if x_s.shape[0] == 0:
            b = False
            continue

        # Move them from the unlabeled set to the train one
        x_l = np.concatenate((x_l, x_s))
        y_l = np.concatenate((y_l, y_s))
        x_u = np.delete(x_u, np.where(selection), axis=0)
        s = x_l.shape[0] - l
        sample_distr = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / s, s)))

        # Stop criterion
        if x_u.shape[0] == 0:
            b = False
    return H, thetas


def fsla(x_l, y_l, x_u, theta, max_iter, **kwargs):
    """
    A margin-based self-learning algorithm.
    :param x_l: Labeled observations.
    :param y_l: Labels.
    :param x_u:  Unlabeled data. Will be used for learning.
    :param theta: Theta
    :param max_iter: A maximum number of iterations that self-learning does.
    :return: The final classification model H that has been trained on (x_l, y_l)
    and pseudo-labeled (x_u, yPred)
    """

    if 'n_estimators' not in kwargs:
        n_est = 200
    else:
        n_est = kwargs['n_estimators']

    if 'random_state' not in kwargs:
        rand_state = None
    else:
        rand_state = kwargs['random_state']

    classifier = RandomForestClassifier(n_estimators=n_est, oob_score=True, n_jobs=-1, random_state=rand_state)
    l = x_l.shape[0]
    sample_distr = np.repeat(1 / l, l)
    n = 1
    b = True
    while b:
        u = x_u.shape[0]
        # Learn a classifier
        H = classifier
        H.fit(x_l, y_l, sample_weight=sample_distr)
        margin_u = H.predict_proba(x_u)
        pred_u = np.argmax(margin_u, axis=1)

        # Select observations with argmax margin more than corresponding theta
        selection = np.array(margin_u[np.arange(u), pred_u] >= theta)
        x_s = x_u[selection, :]
        y_s = pred_u[selection]

        # Move them from the unlabeled set to the train one
        x_l = np.concatenate((x_l, x_s))
        y_l = np.concatenate((y_l, y_s))
        x_u = np.delete(x_u, np.where(selection), axis=0)

        s = x_l.shape[0] - l
        if x_s.shape[0] == 0:
            b = False
            continue
        sample_distr = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / s, s)))

        # Stop criterion
        if x_u.shape[0] == 0:
            b = False
        n += 1
        if n == max_iter:
            b = False
    return H
