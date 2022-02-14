import numpy as np
cimport numpy as np
from cpython cimport array
import array
# from cython.parallel import prange


cdef np.double_t c_joint_bayes_risk(np.ndarray[np.float_t, ndim=2] vote, np.ndarray[np.float_t, ndim=2] prob,
np.ndarray[np.int_t, ndim=1] pred, unsigned int i, unsigned int j, double theta, unsigned int samplingRate = 50):
    cdef:
        double ui
        np.ndarray[np.float_t, ndim=1] gammas
        double infimum
        list upper_bounds = []
        unsigned int u, ub_size
        unsigned int n, t
        # M_ij = Mt_ij - Mg_ij
        double I_ij, K_ij, M_ij, A, upper_bound
    infimum = 1e+15
    u = vote.shape[0]
    ui = 0.0
    for t in range(u):
        ui += prob[t, i]
    # K_ij doesn't depend on gamma, so it is computed in advance
    K_ij = 0.0
    for t in range(u):
        if pred[t] == j:
            K_ij += prob[t, i] * vote[t, j]
    K_ij /= ui
    gammas = np.zeros(samplingRate, dtype=np.float)
    for t in range(samplingRate):
        gammas[t] = theta + (t + 1) * (1 - theta) / samplingRate
    ub_size = 0
    for n in range(samplingRate):
        gamma = gammas[n]
        I_ij = 0.0
        M_ij = 0.0
        for t in range(u):
            if vote[t, j] >= theta and vote[t, j] < gamma:
                I_ij += prob[t, i]
                M_ij += vote[t, j] * prob[t, i]
        I_ij /= ui
        M_ij /= ui
        A = K_ij - M_ij
        upper_bound = I_ij + (A * (A > 0)) / gamma
        upper_bounds.append(upper_bound)
        ub_size += 1
        if upper_bound < infimum:
            infimum = upper_bound
        if n>3:
            if upper_bounds[ub_size-1] > upper_bounds[ub_size-2]:
                if upper_bounds[ub_size-2] >= upper_bounds[ub_size-3]:
                    break
    return infimum


cpdef c_optimal_threshold_vector(np.ndarray[np.float_t, ndim=2] vote, np.ndarray[np.float_t, ndim=2] prob, np.ndarray[np.int_t, ndim=1] pred,
unsigned int K, unsigned int samplingRate = 50):
    cdef:
        list theta = []
        unsigned int u
        unsigned int k, n, i, t, j
        unsigned int num
        list cond_errs = []
        # np.ndarray[np.float_t, ndim=1] cond_errs
        double theta_min
        double theta_max
        double tmp, minim
        double pbl, numerator
        double uk
        np.ndarray[np.int_t, ndim=1] zero_classes
        np.ndarray[np.float_t, ndim=1] thetas
        np.ndarray[np.double_t, ndim=1] count_class
        np.ndarray[np.double_t, ndim=1] confusion_k

    u = vote.shape[0]
    zero_classes = np.zeros(K, dtype=np.int)
    count_class = np.zeros(K, dtype=np.double)
    for k in range(K):
        uk = 0.0
        for t in range(u):
            uk += prob[t, k]
        count_class[k] = uk / u
        if uk == 0:
            zero_classes[k] = 1

    for k in range(K):
        # if the expected number of unlabelled examples from class k is 0
        # then we do not search the optimal theta_k
        if zero_classes[k] == 1:
            theta.append(1.0)
            continue

        theta_min = 1e+08
        theta_max = 0
        for t in range(u):
            if vote[t, k] < theta_min:
                theta_min = vote[t, k]
            if vote[t, k] > theta_max:
                theta_max = vote[t, k]
        # A set of possible thetas:
        thetas = np.zeros(samplingRate, dtype=np.float)
        for t in range(samplingRate):
            thetas[t] = theta_min + t * (theta_max - theta_min) / samplingRate
        cond_errs = []
        # cond_errs = np.zeros(samplingRate, dtype=np.double)
        # for n in prange(samplingRate, nogil=True):
        for n in range(samplingRate):
            # confusion matrix only for class k
            # since the only non-zero column is k, it is stored as a vector
            confusion_k = np.zeros(K)
            for i in range(K):
                if i == k or zero_classes[i] == 1:
                    continue
                else:
                    confusion_k[i] = c_joint_bayes_risk(vote, prob, pred, i, k, thetas[n])

            pbl = 0.0
            for t in range(u):
                if vote[t, k] >= thetas[n] and pred[t] == k:
                    pbl = pbl + 1
            pbl = pbl / u

            if pbl == 0:
                pbl = 1e-8

            numerator = 0.0
            for j in range(K):
                numerator = numerator + count_class[j] * confusion_k[j]
            numerator = numerator / u
            # cond_errs[n] = numerator / pbl
            cond_err = numerator / pbl
            cond_errs.append(cond_err)
            if n > 3:
               if cond_errs[-1] > cond_errs[-2] >= cond_errs[-3]:
                   break
        minim = 1e+15
        for t in range(len(cond_errs)):
            if cond_errs[t] < minim:
                minim = cond_errs[t]
                num = t
        theta.append(thetas[num])
    return np.array(theta)
