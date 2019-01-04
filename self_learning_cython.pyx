import numpy as np
cimport numpy as np
from cpython cimport array
import array
from time import time


cdef np.double_t c_joint_bayes_risk(np.ndarray[np.float_t, ndim=2] margin, np.ndarray[np.int_t, ndim=1] pred, unsigned int i, unsigned int j, double theta, unsigned int samplingRate = 50):
    cdef:
        double ui
        np.ndarray[np.float_t, ndim=1] margins
        np.ndarray[np.float_t, ndim=1] gammas
        double infimum
        list upperBounds = []
        unsigned int u, ub_size
        unsigned int n, t
        # M_ij = Mt_ij - Mg_ij
        double I_ij, K_ij, M_ij, A, upperBound
    infimum = 1e+15
    u = margin.shape[0]
    ui = 0.0
    for t in range(u):
        ui += margin[t, i]
    margins = margin[:, j]
    gammas = np.zeros(samplingRate, dtype=np.float)
    for t in range(samplingRate):
        gammas[t] = theta + (t + 1) * (1 - theta) / samplingRate
    ub_size = 0
    for n in range(samplingRate):
        gamma = gammas[n]
        I_ij = 0.0
        K_ij = 0.0
        M_ij = 0.0
        for t in range(u):
            if pred[t] == j:
                K_ij += margin[t, i] * margins[t]
            if margins[t] >= theta and margins[t] < gamma:
                I_ij += margin[t, i]
                M_ij += margins[t] * margin[t, i]
        I_ij /= ui
        K_ij /= ui
        M_ij /= ui
        A = K_ij - M_ij
        upperBound = I_ij + (A * (A > 0)) / gamma
        upperBounds.append(upperBound)
        ub_size += 1
        if upperBound < infimum:
            infimum = upperBound
        if n>3:
            if upperBounds[ub_size-1] > upperBounds[ub_size-2]:
                if upperBounds[ub_size-2] >= upperBounds[ub_size-3]:
                    break
    return infimum


cpdef c_optimal_threshold_vector(np.ndarray[np.float_t, ndim=2] margin, np.ndarray[np.int_t, ndim=1] pred, unsigned int K, unsigned int samplingRate = 50):
    cdef:
        list theta = []
        unsigned int u
        unsigned int k, n, i, t, j
        unsigned int num
        list BE = []
        double theta_min
        double theta_max
        double tmp, minim
        double pbl, numerator
        double uk
        np.ndarray[np.int_t, ndim=1] zero_classes
        np.ndarray[np.float_t, ndim=1] thetas
        np.ndarray[np.double_t, ndim=1] countClass
        np.ndarray[np.double_t, ndim=1] confusion_k
        np.ndarray[np.float_t, ndim=1] margins

    u = margin.shape[0]
    zero_classes = np.zeros(K, dtype=np.int)
    for k in range(K):
        uk = 0.0
        for t in range(u):
            uk += margin[t, k]
        if uk == 0:
            zero_classes[k] = 1

    for k in range(K):
        # if the expected number of unlabelled examples from class k is 0
        # then we do not search the optimal theta_k
        if zero_classes[k] == 1:
            theta.append(1.0)
            continue

        margins = margin[:, k]
        theta_min = 1e+08
        theta_max = 0
        for t in range(margins.shape[0]):
            if margins[t] < theta_min:
                theta_min = margins[t]
            if margins[t] > theta_max:
                theta_max = margins[t]
        # A set of possible thetas:
        thetas = np.zeros(samplingRate, dtype=np.float)
        for t in range(samplingRate):
            thetas[t] = theta_min + t * (theta_max - theta_min) / samplingRate
        BE = []
        for n in range(samplingRate):
            # confusion matrix only for class k
            # since the only non-zero column is k, it is stored as a vector
            confusion_k = np.zeros(K)
            for i in range(K):
                if i == k or zero_classes[i] == 1:
                    continue
                else:
                    confusion_k[i] = c_joint_bayes_risk(margin, pred, i, k, thetas[n])

            pbl = 0.0
            for t in range(u):
                if margin[t, k] >= thetas[n] and pred[t] == k:
                    pbl += 1
            pbl = pbl / u

            if pbl == 0:
                pbl = 1e-8

            countClass = np.zeros(K)
            for j in range(K):
                tmp = 0.0
                for t in range(u):
                    tmp += margin[t, j]
                countClass[j] = tmp
            confusion_k
            numerator = 0.0
            for j in range(K):
                numerator += countClass[j] * confusion_k[j]
            numerator = numerator / u
            kek = numerator / pbl
            BE.append(kek)
            if n>3:
                if BE[-1]>BE[-2] and BE[-2]>=BE[-3]:
                    break
        minim = 1e+15
        for t in range(len(BE)):
            if BE[t] < minim:
                minim = BE[t]
                num = t
        theta.append(thetas[num])
    return np.array(theta)
