import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from itertools import product
from utility import calc_mse

class KernelRegression(BaseEstimator, RegressorMixin):

    def __init__(self, kernel="rbf", gamma=None):
        self.kernel = kernel
        self.gamma = gamma
        self.val_set = None

    def get_params(self):
        return {'gamma': self.gamma}

    def set_params(self, params):
        self.gamma = params['gamma']
        

    def set_val(self, val_set, val_labels):
        self.val_set = val_set
        self.val_labels = val_labels

    def fit(self, X, y):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.y = y

        if hasattr(self.gamma, "__iter__"):
            self.gamma = self.find_opt_gamma()

        return self

    def predict(self, X):
        K = pairwise_kernels(self.X, self.scaler.transform(X), metric=self.kernel, gamma=self.gamma)
        return np.nan_to_num((K * self.y[:, np.newaxis]).sum(axis=0) / K.sum(axis=0), False)

    def find_opt_gamma(self):
        mse = np.empty_like(self.gamma, dtype=np.float)
        if self.val_set is not None:
            cv_x = self.X
            cv_y = self.y
            val_x = self.val_set
            val_y = self.val_labels
        else:
            idx = np.arange(len(self.X))
            val_size = int(0.2 * len(self.X))
            rng = np.random.default_rng()
            rng.shuffle(idx, 0)
            val_idx = idx[:val_size]
            train_idx = idx[val_size:]
            cv_x = self.X[train_idx]
            cv_y = self.y[train_idx]
            val_x = self.X[val_idx]
            val_y = self.y[val_idx]
        for i, gamma in enumerate(self.gamma):
            K = pairwise_kernels(cv_x, self.scaler.transform(val_x), metric=self.kernel, gamma=gamma)
            Ky = K * cv_y[:, np.newaxis]
            y_pred = Ky.sum(axis=0) / K.sum(axis=0)
            mse[i] = np.mean((y_pred - val_y) ** 2)
        if np.all(np.isnan(mse)):
            print('All NaN in NW Reg!')
            return 1.0
        return self.gamma[np.nanargmin(mse)]

class MyForest():
    def __init__(self, trees, depth, leaf_samples, max_features = None) -> None:
        self.trees = trees
        self.depth = depth
        self.leaf_samples = leaf_samples
        self.max_features = max_features
        self.val_set = None
        self.val_labels = None
        self.params = None

    def set_val(self, val_set, val_labels):
        self.val_set = val_set
        self.val_labels = val_labels

    def get_params(self):
        return {'n_estimators': self.n_estims, 'max_depth': self.max_depth, 'min_samples_leaf': self.min_samples_leaf, 'max_features': self.max_features}

    def set_params(self, params):
        self.params = params

    def __call__(self):
        return self

    def fit(self, x, y):
        if self.params is not None:
            self.tree = RandomForestRegressor(**self.params)
            self.tree.fit(x, y)
            return self
        if self.val_set is not None and self.val_labels is not None:
            min_mse = 10 ** 9
            for n_estimators, max_depth, min_samples_leaf in product(self.trees, self.depth, self.leaf_samples):
                tree = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=self.max_features)
                tree.fit(x, y)
                cur_mse = calc_mse(tree.predict(self.val_set), self.val_labels)
                if cur_mse < min_mse:
                    min_mse = cur_mse
                    self.tree = tree
                    self.n_estims = n_estimators
                    self.max_depth = max_depth
                    self.min_samples_leaf = min_samples_leaf
        else:
            # print('No validation set was provided!')
            self.tree = RandomForestRegressor()
            self.tree.fit(x, y)

    def predict(self, x):
        return self.tree.predict(x)