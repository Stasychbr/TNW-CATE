import numpy as np
 
class BaseLearner():
    def __init__(self, estimator_class) -> None:
        if callable(getattr(estimator_class, 'fit')) \
            and callable(getattr(estimator_class, 'predict')) \
            and hasattr(estimator_class, '__init__'):
            self.estimator_class = estimator_class
            self.is_fitted = False
        else:
            raise 'estimator_class must be an object type and have fit and predict methods'


class TLearner(BaseLearner):
    def __init__(self, estimator_class, *estim_args, **estim_kwargs) -> None:
        super().__init__(estimator_class)
        self.estim_control = self.estimator_class(*estim_args, **estim_kwargs)
        self.estim_treat = self.estimator_class(*estim_args, **estim_kwargs)

    def fit(self, X, Y, W):
        control_idx = W == 0
        X_train_control = X[control_idx]
        Y_train_control = Y[control_idx]
        self.estim_control.fit(X_train_control, Y_train_control)
        treat_idx = W == 1
        X_train_treat = X[treat_idx]
        Y_train_treat = Y[treat_idx]
        self.estim_treat.fit(X_train_treat, Y_train_treat)
        self.is_fitted = True
    
    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X) - self.estim_control.predict(X)

    def predict_control(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_control.predict(X)

    def predict_treat(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X)

class SLearner(BaseLearner):
    def __init__(self, estimator_class, *estim_args, **estim_kwargs) -> None:
        super().__init__(estimator_class)
        self.estim = self.estimator_class(*estim_args, **estim_kwargs)

    def fit(self, X, Y, W):
        X_train = np.concatenate((X, W[:, np.newaxis]), axis=1)
        self.estim.fit(X_train, Y)
        self.is_fitted = True
    
    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat) - self.estim.predict(X_control)

    def predict_control(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_control)

    def predict_treat(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat)

class XLearner(BaseLearner):
    def __init__(self, estimator_class, *estim_args, **estim_kwargs) -> None:
        super().__init__(estimator_class)
        self.mu_0 = self.estimator_class(*estim_args, **estim_kwargs)
        self.mu_1 = self.estimator_class(*estim_args, **estim_kwargs)
        self.tau_0 = self.estimator_class(*estim_args, **estim_kwargs)
        self.tau_1 = self.estimator_class(*estim_args, **estim_kwargs)

    def fit(self, X, Y, W):
        self.g = np.sum(W) / W.shape[0]
        
        cnt_idx = W == 0
        X_0, Y_0 = X[cnt_idx], Y[cnt_idx]
        tr_idx = W == 1
        X_1, Y_1 = X[tr_idx], Y[tr_idx]
        
        self.mu_0.fit(X_0, Y_0)
        self.mu_1.fit(X_1, Y_1)
        
        D_0 = self.mu_1.predict(X_0) - Y_0
        D_1 = Y_1 - self.mu_0.predict(X_1)

        self.tau_0.fit(X_0, D_0)
        self.tau_1.fit(X_1, D_1)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.g * self.tau_0.predict(X) + (1 - self.g) * self.tau_1.predict(X)



class TValLearner():
    def __init__(self, estimator1, estimator2, val_set_c, val_labels_c, val_set_t, val_labels_t) -> None:
        self.estim_control = estimator1
        self.estim_treat = estimator2
        self.val_set_c = val_set_c
        self.val_labels_c = val_labels_c
        self.val_set_t = val_set_t
        self.val_labels_t = val_labels_t

    def fit(self, X, Y, W):
        control_idx = W == 0
        X_train_control = X[control_idx]
        Y_train_control = Y[control_idx]
        self.estim_control.set_val(self.val_set_c, self.val_labels_c)
        self.estim_control.fit(X_train_control, Y_train_control)
        treat_idx = W == 1
        X_train_treat = X[treat_idx]
        Y_train_treat = Y[treat_idx]
        # self.estim_treat.set_val(self.val_set_t, self.val_labels_t)
        self.estim_treat.set_params(self.estim_control.get_params())
        self.estim_treat.fit(X_train_treat, Y_train_treat)
        self.is_fitted = True
    
    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X) - self.estim_control.predict(X)

    def predict_control(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_control.predict(X)

    def predict_treat(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.estim_treat.predict(X)

class SValLearner():
    def __init__(self, estimator1, val_set_c, val_labels_c, val_set_t, val_labels_t) -> None:
        self.estim = estimator1
        self.val_set_c = np.concatenate((val_set_c, np.zeros((val_set_c.shape[0], 1))), axis=1)
        self.val_labels_c = val_labels_c
        # self.val_set_t = np.concatenate((val_set_t, np.ones((val_set_t.shape[0], 1))), axis=1)
        # self.val_labels_t = val_labels_t

    def fit(self, X, Y, W):
        X_train = np.concatenate((X, W[:, np.newaxis]), axis=1)
        self.estim.set_val(self.val_set_c, self.val_labels_c)
        self.estim.fit(X_train, Y)
        self.is_fitted = True
    
    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat) - self.estim.predict(X_control)

    def predict_control(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_control = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_control)

    def predict_treat(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        X_treat = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return self.estim.predict(X_treat)

class XValLearner():
    def __init__(self, estimator1, estimator2, estimator3, estimator4, val_set_c, val_labels_c, val_set_t, val_labels_t) -> None:
        self.mu_0 = estimator1
        self.mu_1 = estimator2
        self.tau_0 = estimator3
        self.tau_1 = estimator4
        self.val_set_c = val_set_c
        self.val_labels_c = val_labels_c
        self.val_set_t = val_set_t
        self.val_labels_t = val_labels_t

    def fit(self, X, Y, W):
        self.g = np.sum(W) / W.shape[0]
        
        cnt_idx = W == 0
        X_0, Y_0 = X[cnt_idx], Y[cnt_idx]
        tr_idx = W == 1
        X_1, Y_1 = X[tr_idx], Y[tr_idx]
        
        self.mu_0.set_val(self.val_set_c, self.val_labels_c)
        self.mu_0.fit(X_0, Y_0)
        # self.mu_1.set_val(self.val_set_t, self.val_labels_t)
        self.mu_1.set_params(self.mu_0.get_params())
        self.mu_1.fit(X_1, Y_1)
        
        D_0 = self.mu_1.predict(X_0) - Y_0
        D_1 = Y_1 - self.mu_0.predict(X_1)
        
        # self.tau_0.set_val(self.val_set_c, self.mu_1.predict(self.val_set_c) - self.val_labels_c)
        self.tau_0.set_params(self.mu_0.get_params())
        self.tau_0.fit(X_0, D_0)
        # self.tau_1.set_val(self.val_set_t, self.mu_0.predict(self.val_set_t) - self.val_labels_t)
        self.tau_1.set_params(self.mu_0.get_params())
        self.tau_1.fit(X_1, D_1)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise 'model must be fitted first'
        return self.g * self.tau_0.predict(X) + (1 - self.g) * self.tau_1.predict(X)

def make_t_learner(val_set_c, val_labels_c, val_set_t, val_labels_t, base, *args, **estim_kwargs):
    tree1 = base(*args, **estim_kwargs) # MyForest(trees, depth, leaf_samples)
    tree2 = base(*args, **estim_kwargs) # MyForest(trees, depth, leaf_samples)
    return TValLearner(tree1, tree2, val_set_c, val_labels_c, val_set_t, val_labels_t)

def make_s_learner(val_set_c, val_labels_c, val_set_t, val_labels_t, base, *args, **estim_kwargs):
    tree = base(*args, **estim_kwargs) # MyForest(trees, depth, leaf_samples)
    return SValLearner(tree, val_set_c, val_labels_c, val_set_t, val_labels_t)

def make_x_learner(val_set_c, val_labels_c, val_set_t, val_labels_t, base, *args, **estim_kwargs):
    tree1 = base(*args, **estim_kwargs) # MyForest(trees, depth, leaf_samples)
    tree2 = base(*args, **estim_kwargs) # MyForest(trees, depth, leaf_samples)
    tree3 = base(*args, **estim_kwargs) # MyForest(trees, depth, leaf_samples)
    tree4 = base(*args, **estim_kwargs) # MyForest(trees, depth, leaf_samples)
    return XValLearner(tree1, tree2, tree3, tree4, val_set_c, val_labels_c, val_set_t, val_labels_t)