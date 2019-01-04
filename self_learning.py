import numpy as np
from sklearn.ensemble import RandomForestClassifier
# import pyximport;
# pyximport.install()
import self_learning_cython as slc


def joint_bayes_risk(margin, pred, i, j, theta, samplingRate=50):
    # li = \sum_{x\in X_U} \I{y=i} =approx.= \sum_{x\in X_U} m_Q(x,i)
    li = np.sum(margin[:, i])
    margins = margin[:, j]
    # gammas = sorted(list(set(margins[margins > theta])))
    gammas = theta + (1 - theta) * (np.arange(samplingRate) + 1) / samplingRate
    infimum = 1e+05
    gamma_star = 0
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


def optimal_threshold_vector(margin, pred, K, printy=True, samplingRate=50):
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
                    matrix[i, k] = joint_bayes_risk(margin, pred, i, k, thetas[n], printy)
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


def msla(xTrain, yTrain, xTest, cython=True, **kwargs):
    """
    A margin-based self-learning algorithm.
    :param xTrain: Labeled observations.
    :param yTrain: Labels.
    :param xTest:  Unlabeled data. Will be used for learning.
    :param cython:  Whether or not to use cython code, which gives speedup in computation. The default value is True.
    :return: The final classification model H that has been trained on (xTrain, yTrain)
    and pseudo-labeled (xTest, yPred)
    """

    if 'n_estimators' not in kwargs:
        n_est = 200
    else:
        n_est = kwargs['n_estimators']

    classifier = RandomForestClassifier(n_estimators=n_est, oob_score=True, n_jobs=-1)
    l = xTrain.shape[0]
    sample_distr = np.repeat(1 / l, l)
    K = np.unique(yTrain).shape[0]
    b = True
    thetas = []
    while b:
        u = xTest.shape[0]
        # Learn a classifier
        H = classifier
        H.fit(xTrain, yTrain, sample_weight=sample_distr)
        marginTest = H.predict_proba(xTest)
        predTest = np.argmax(marginTest, axis=1)

        # Find a threshold minimizing Bayes conditional error
        if cython:
            theta = slc.c_optimal_threshold_vector(marginTest, predTest, K)
        else:
            theta = optimal_threshold_vector(marginTest, predTest, K)
        thetas.append(theta)

        # Select observations with argmax margin more than corresponding theta
        selection = np.array(marginTest[np.arange(u), predTest] >= theta[predTest])
        xS = xTest[selection, :]
        yS = predTest[selection]
        # Stop if there is no anything to add:
        if xS.shape[0] == 0:
            b = False
            continue

        # Move them from the test set to the train one
        xTrain = np.concatenate((xTrain, xS))
        yTrain = np.concatenate((yTrain, yS))
        xTest = np.delete(xTest, np.where(selection), axis=0)
        s = xTrain.shape[0] - l
        sample_distr = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / s, s)))

        # Stop criterion
        if xTest.shape[0] == 0:
            b = False
    return H, thetas


def fsla(xTrain, yTrain, xTest, theta, max_iter, **kwargs):
    """
    A margin-based self-learning algorithm.
    :param xTrain: Labeled observations.
    :param yTrain: Labels.
    :param xTest:  Unlabeled data. Will be used for learning.
    :param theta: Theta
    :return: The final classification model H that has been trained on (xTrain, yTrain)
    and pseudo-labeled (xTest, yPred)
    """

    if 'n_estimators' not in kwargs:
        n_est = 200
    else:
        n_est = kwargs['n_estimators']

    classifier = RandomForestClassifier(n_estimators=n_est, oob_score=True, n_jobs=-1)
    l = xTrain.shape[0]
    sample_distr = np.repeat(1 / l, l)
    n = 1
    b = True
    while b:
        u = xTest.shape[0]
        # Learn a classifier
        H = classifier
        H.fit(xTrain, yTrain, sample_weight=sample_distr)
        marginTest = H.predict_proba(xTest)
        predTest = np.argmax(marginTest, axis=1)

        # Select observations with argmax margin more than corresponding theta
        selection = np.array(marginTest[np.arange(u), predTest] >= theta)
        xS = xTest[selection, :]
        yS = predTest[selection]

        # Move them from the test set to the train one
        xTrain = np.concatenate((xTrain, xS))
        yTrain = np.concatenate((yTrain, yS))
        xTest = np.delete(xTest, np.where(selection), axis=0)

        s = xTrain.shape[0] - l
        if xS.shape[0] == 0:
            b = False
            continue
        sample_distr = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / s, s)))

        # Stop criterion
        if xTest.shape[0] == 0:
            b = False
        n += 1
        if n == max_iter:
            b = False
    return H
