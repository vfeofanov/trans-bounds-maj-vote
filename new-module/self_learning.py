from .pseudo_label_policy import *
import numpy as np
from copy import deepcopy


class SelfLearning:
    def __init__(self, base_estimator=None, policy='confidence', voting='soft', theta='auto', cython=True, 
                 sup_prob=True, worst_prob=False, fixed_prob=None, num_take=None, decreased_pl_weights=True, 
                 max_iter=None, restart=False, curriculum_step=0.2, semisup_base=False, random_state=None):
        # TODO: option to enforce pseudo-labeling all unlab. examples; shuffle unlab. examples back
        self.policy = policy
        self.num_take = num_take
        # TODO: voting is not used
        self.voting = voting
        self.theta = theta
        self.cython = cython
        self.sup_prob = sup_prob
        self.worst_prob = worst_prob
        self.fixed_prob = fixed_prob
        self.max_iter = max_iter
        if self.max_iter is not None:
            if type(self.max_iter) != int:
                raise KeyError("max_iter must be either None or a strictly positive integer")
            elif self.max_iter < 1:
                raise KeyError("max_iter must be either None or a strictly positive integer")
        self.restart = restart
        self.curriculum_step = curriculum_step
        if self.restart and self.max_iter is None and "cur" not in self.theta:
            raise KeyError("max_iter must be set to a strictly positive integer if restart is True "
                           "and if it is not curriculum learning")
        self._initialize_policy_()
        self.decreased_pl_weights = decreased_pl_weights
        self.semisup_base = semisup_base
        self.random_state = random_state
        # validate base_estimator
        self.base_estimator = base_estimator
        self._validate_base_estimator()
        # initialize self.base_estimator_
        self._initialize_base_estimator_()
        self.init_classifier = None
        self.final_classifier = None
        self.log_all_iterations = None
        self.x_pl = None
        self.y_pl = None

    def fit(self, x_l, y_l, x_u):
        """
        :param x_l: Labeled training observations
        :param y_l: Labels
        :param x_u:  Unlabeled training observations
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        l = x_l.shape[0]
        u = x_u.shape[0]
        # initialize log to store basic info about each iteration of self-learning
        log = LogStorer(x_l, y_l, x_u)
        cond = True
        it = 0
        idx_u = np.arange(u)
        sample_weight = None
        # initialization: supervised model trained on labeled examples only
        estimator = deepcopy(self.base_estimator_)
        if self.semisup_base:
            estimator.fit(x_l, y_l, x_u)
        else:
            estimator.fit(x_l, y_l)
        log.update(estimator, x_l, y_l, x_u, [], None)
        # start of self-learning
        while cond:
            it += 1
            # choose examples to pseudo-label
            self.policy_.choose(log=log)
            selection = self.policy_.returned_subset
            # stop if there is no anything to add:
            if selection.size == 0:
                cond = False
                continue
            # select the examples and pseudo-label them
            x_s = x_u[selection, :]
            y_s = estimator.predict(x_s)
            idx_s = idx_u[selection]
            if self.restart:
                # move them from the unlabeled set to the labeled one
                x_l = np.concatenate((x_l[:l], x_s))
                y_l = np.concatenate((y_l[:l], y_s))
                idx_pl = idx_s
            else:
                # move them from the unlabeled set to the labeled one
                x_l = np.concatenate((x_l, x_s))
                y_l = np.concatenate((y_l, y_s))
                idx_pl = np.concatenate((log.current_iteration['idx_pl'], idx_s))
                x_u = np.delete(x_u, selection, axis=0)
                idx_u = np.delete(idx_u, selection)
            if self.theta is None:
                theta = None
            else:
                theta = self.policy_.theta
            # if True, the weight of pseudo-labeled examples is decreased
            if self.decreased_pl_weights:
                u_pl = x_l.shape[0] - l
                sample_weight = np.concatenate((np.repeat(1 / l, l), np.repeat(1 / u_pl, u_pl)))

            # learn a new classifier
            estimator = deepcopy(self.base_estimator_)
            if self.semisup_base:
                estimator.fit(x_l, y_l, x_u, sample_weight=sample_weight)
            else:
                if sample_weight is None:
                    estimator.fit(x_l, y_l)
                else:
                    estimator.fit(x_l, y_l, sample_weight=sample_weight)
            # update log
            log.update(estimator, x_l, y_l, x_u, idx_pl, theta)
            # stop if max_iter is reached
            if it == self.max_iter:
                cond = False
            # stop if all unlabeled examples are pseudo-labeled
            if x_l.shape[0] == l + u:
                cond = False
        self.init_classifier = log.previous_iterations[0]['estimator']
        self.final_classifier = log.current_iteration['estimator']
        log_all_iterations = log.previous_iterations
        log_all_iterations.append(log.current_iteration)
        self.log_all_iterations = log_all_iterations
        self.x_pl = log.current_iteration['x_l']
        self.y_pl = log.current_iteration['y_l']

    def predict(self, x, supervised=False):
        if supervised:
            return self.init_classifier.predict(x)
        else:
            return self.final_classifier.predict(x)

    def predict_proba(self, x, supervised=False):
        if supervised:
            return self.init_classifier.predict_proba(x)
        else:
            return self.final_classifier.predict_proba(x)

    def _validate_base_estimator(self):
        if self.base_estimator is not None:
            base_estimator_methods = ['fit', 'predict']
            if self.policy == 'confidence':
                base_estimator_methods.append('predict_proba')
            if self.theta != 'auto':
                if not np.all(list(map(lambda method: hasattr(self.base_estimator, method), base_estimator_methods))):
                    raise KeyError("base_estimator doesn't contain one or any of the following methods: " +
                                   ", ".join(base_estimator_methods))
            else:
                if not np.all(list(map(lambda method: hasattr(self.base_estimator, method), base_estimator_methods))):
                    raise KeyError("base_estimator doesn't contain one or any of the following methods: " +
                                   ", ".join(base_estimator_methods))

    def _initialize_policy_(self):
        if self.policy == 'confidence':
            if self.num_take is not None:
                self.policy_ = MostConfident(num_take=self.num_take)
                # theta is enforced to be None if policy_ is MostConfident
                self.theta = None
            else:
                if self.theta == 'mean':
                    self.policy_ = MeanPredictionVote()
                elif self.theta == 'auto':
                    self.policy_ = TransductiveConditionalError(cython=self.cython, sup_prob=self.sup_prob,
                                                                worst_prob=self.worst_prob, fixed_prob=self.fixed_prob)
                elif self.theta == 'curriculum':
                    self.policy_ = CurriculumLearning(curriculum_step=self.curriculum_step)
                else:
                    self.policy_ = ConfidenceThreshold(theta=self.theta)
        elif self.policy == 'random':
            self.policy_ = RandomSubset(num_take=self.num_take)
            # theta is enforced to be None if policy_ is RandomSubset
            self.theta = None
        else:
            # TODO: possibility of custom policy
            raise KeyError("policy must be either confidence or random.")

    def _initialize_base_estimator_(self):
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
            self._agree_random_state()
        # by default, base_estimator_ is a random forest
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.base_estimator_ = RandomForestClassifier(random_state=self.random_state)

    def _agree_random_state(self):
        # if base_estimator_ has random_state attribute
        if hasattr(self.base_estimator_, 'random_state'):
            # and if random_state is not default,
            # replace the rs of base_estimator_ by random_state
            if self.random_state is not None:
                # we raise the warning, if the rd of base_estimator_ is not None initially
                if self.random_state != self.base_estimator_.random_state is not None:
                    raise Warning("random state of base_estimator_ is set to " + str(self.random_state))
                self.base_estimator_.random_state = self.random_state


class LogStorer:
    def __init__(self, x_l, y_l, x_u):
        self.x_l = x_l
        self.y_l = y_l
        self.x_u = x_u
        self.current_iteration = None
        self.previous_iterations = list()
        self.idx_iter = 0

    def update(self, estimator, x_l, y_l, x_u, idx_pl, theta=None):
        if self.current_iteration is not None:
            self.previous_iterations.append(self.current_iteration)
        self.current_iteration = {
            "estimator": estimator,
            "x_l": x_l,
            "y_l": y_l,
            "x_u": x_u,
            "idx_pl": idx_pl,
            "theta": theta,
        }
        self.idx_iter += 1
