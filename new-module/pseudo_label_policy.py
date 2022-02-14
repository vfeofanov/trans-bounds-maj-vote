import numpy as np
# import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()},
#                                     language_level="3", reload_support=True)
from .self_learning_cython import c_optimal_threshold_vector


class Policy:
    def __init__(self):
        self.returned_subset = np.array([])

    def choose(self, log):
        return self.returned_subset


class RandomSubset(Policy):
    def __init__(self, num_take=0.1):
        super().__init__()
        self.num_take = num_take
        self._validate_num_take()

    def _validate_num_take(self):
        if np.isreal(self.num_take):
            if self.num_take <= 0:
                raise KeyError("The parameter num_take must be a positive number.")
            if 1 <= self.num_take != int(self.num_take):
                raise KeyError("The parameter num_take must be either a float within [0,1) or an integer >= 1.")
        else:
            raise KeyError("The parameter num_take must be either a float within [0,1) or an integer >= 1.")

    def choose(self, log):
        x_u = log.current_iteration["x_u"]
        u = x_u.shape[0]
        if type(self.num_take) == float:
            n_take = int(self.num_take * u)
        else:
            n_take = self.num_take
        self.returned_subset = np.random.choice(np.arange(u), n_take, replace=False)


class MostConfident(Policy):
    def __init__(self, num_take=0.1):
        super().__init__()
        self.num_take = num_take
        self._validate_num_take()

    def _validate_num_take(self):
        if np.isreal(self.num_take):
            if self.num_take <= 0:
                raise KeyError("The parameter num_take must be a positive number.")
            if 1 <= self.num_take != int(self.num_take):
                raise KeyError("The parameter num_take must be either a float within [0,1) or an integer >= 1.")
        else:
            raise KeyError("The parameter num_take must be either a float within [0,1) or an integer >= 1.")

    def choose(self, log):
        x_u = log.current_iteration["x_u"]
        estimator = log.current_iteration["estimator"]
        prediction_vote_u = estimator.predict_proba(x_u).max(axis=1)
        u = x_u.shape[0]
        if type(self.num_take) == float:
            n_take = int(self.num_take * u)
        else:
            n_take = self.num_take
        self.returned_subset = np.argsort(prediction_vote_u)[::-1][:n_take]


class ConfidenceThreshold(Policy):
    def __init__(self, theta=None):
        super().__init__()
        if theta is None:
            self.theta = 1
        else:
            self.theta = theta

    def choose(self, log):
        y_l = log.current_iteration["y_l"]
        K = np.unique(y_l).size
        x_u = log.current_iteration["x_u"]
        estimator = log.current_iteration["estimator"]
        vote_u = estimator.predict_proba(x_u)
        y_pred_u = vote_u.argmax(axis=1)
        u = x_u.shape[0]
        is_theta_vector = self._validate_theta(K)
        if is_theta_vector:
            self.returned_subset = np.where(vote_u[np.arange(u), y_pred_u] >= self.theta[y_pred_u])[0]
        else:
            self.returned_subset = np.where(vote_u[np.arange(u), y_pred_u] >= self.theta)[0]

    def _validate_theta(self, K):
        if type(self.theta) in [float, int]:
            if not (0 <= self.theta <= 1):
                raise KeyError("theta's value(s) must lie within the interval [0,1].")
            else:
                return False
        elif type(self.theta) == np.ndarray or type(self.theta) == list:
            self.theta = np.array(self.theta)
            if not np.all(np.logical_and(self.theta >= 0, self.theta <= 1)):
                raise KeyError("theta's value(s) must lie within the interval [0,1].")
            elif self.theta.size == K:
                return True
            else:
                raise KeyError("theta must be either a scalar or a vector of size K.")


class MeanPredictionVote(Policy):
    def __init__(self):
        super().__init__()
        self.theta = None

    def choose(self, log):
        x_u = log.current_iteration["x_u"]
        estimator = log.current_iteration["estimator"]
        prediction_vote_u = estimator.predict_proba(x_u).max(axis=1)
        self.theta = np.mean(prediction_vote_u)
        base_policy = ConfidenceThreshold(theta=self.theta)
        base_policy.choose(log)
        self.returned_subset = base_policy.returned_subset


class TransductiveConditionalError(Policy):
    def __init__(self, cython=True, sup_prob=False, worst_prob=False, fixed_prob=None):
        super().__init__()
        self.cython = cython
        self.sup_prob = sup_prob
        self.worst_prob = worst_prob
        self.fixed_prob = fixed_prob
        self.theta = None

    def choose(self, log):
        x_u = log.current_iteration["x_u"]
        x_l = log.current_iteration["x_l"]
        y_l = log.current_iteration["y_l"]
        estimator = log.current_iteration["estimator"]
        K = np.unique(y_l).size
        vote_u = estimator.predict_proba(x_u)
        if self.sup_prob:
            if len(log.previous_iterations) == 0:
                prob_u = vote_u
            else:
                prob_u = log.previous_iterations[0]['estimator'].predict_proba(x_u)
        elif self.worst_prob:
            prob_u = np.full(vote_u.shape, 1/K)
        elif self.fixed_prob is not None:
            prob_u = self.fixed_prob
        else:
            prob_u = vote_u

        y_pred_u = vote_u.argmax(axis=1)
        if self.cython:
            self.theta = c_optimal_threshold_vector(vote_u, prob_u, y_pred_u, K)
        else:
            self.theta = _optimal_threshold_vector(vote_u, prob_u, y_pred_u, K)
        base_policy = ConfidenceThreshold(theta=self.theta)
        base_policy.choose(log)
        self.returned_subset = base_policy.returned_subset


class CurriculumLearning(Policy):
    def __init__(self, curriculum_step=0.2):
        super().__init__()
        self.curriculum_step = curriculum_step
        self.theta = None

    def choose(self, log):
        x_u = log.current_iteration["x_u"]
        estimator = log.current_iteration["estimator"]
        vote_u = estimator.predict_proba(x_u)
        quantile = 1 - (log.idx_iter + 1) * self.curriculum_step
        if quantile < self.curriculum_step:
            quantile = 0
        self.theta = np.quantile(vote_u.max(axis=1), quantile)
        base_policy = ConfidenceThreshold(theta=self.theta)
        base_policy.choose(log)
        self.returned_subset = base_policy.returned_subset


def _joint_bayes_risk(vote, prob, pred, i, j, theta, sampling_rate=50):
    # ui = \sum_{x\in X_U} \I{y=i} =approx.= \sum_{x\in X_U} m_Q(x,i)
    ui = np.sum(vote[:, i])
    vote_j = vote[:, j]
    prob_i = prob[:, i]
    # gammas = sorted(list(set(vote_j[vote_j > theta])))
    gammas = theta + (1 - theta) * (np.arange(sampling_rate) + 1) / sampling_rate
    infimum = 1e+05
    upper_bounds = []
    # for gamma in gammas:
    for n in range(np.size(gammas)):
        gamma = gammas[n]
        I_ij = np.sum(prob_i[np.array((vote_j < gamma) & (vote_j >= theta))]) / ui
        K_ij = np.dot(prob_i, np.array(pred == j) * vote_j) / ui
        # M-less of gamma
        Mg_ij = np.dot(prob_i, np.array(vote_j < gamma) * vote_j) / ui
        # M-less of theta
        Mt_ij = np.dot(prob_i, np.array(vote_j < theta) * vote_j) / ui
        A = K_ij + Mt_ij - Mg_ij
        upper_bound = I_ij + (A * (A > 0)) / gamma
        upper_bounds.append(upper_bound)
        if upper_bound < infimum:
            infimum = upper_bound
        if n > 3:
            if upper_bounds[-1] > upper_bounds[-2] >= upper_bounds[-3]:
                break
    return infimum


def _optimal_threshold_vector(vote, prob, pred, K, sampling_rate=50):
    theta = []

    def reduction(matrix, prob):
        u, K = prob.shape
        count_class = np.array([np.sum(prob[:, j]) for j in range(K)])
        return (1 / u) * np.dot(count_class, np.sum(matrix, axis=1))

    u = vote.shape[0]
    for k in range(K):
        # A set of possible thetas:
        theta_min = np.min(vote[:, k])
        theta_max = np.max(vote[:, k])
        thetas = theta_min + np.arange(sampling_rate) * (theta_max - theta_min) / sampling_rate
        JBR = []
        BE = []
        for n in range(sampling_rate):
            matrix = np.zeros((K, K))
            for i in range(K):
                if i == k:
                    continue
                else:
                    matrix[i, k] = _joint_bayes_risk(vote, prob, pred, i, k, thetas[n])
                    if (i == 0) and (k == 1):
                        JBR.append(matrix[i, k])

            pbl = (1 / u) * np.sum((vote[:, k] >= thetas[n]) & (pred == k))
            if pbl == 0:
                pbl = 1e-15
            BE.append(reduction(matrix, prob)/pbl)
            if n > 3:
                if BE[-1] > BE[-2] >= BE[-3]:
                    break
        BE = np.array(BE)
        num = np.argmin(BE)
        if type(num) is list:
            num = num[0]
        theta.append(thetas[num])
    return np.array(theta)
