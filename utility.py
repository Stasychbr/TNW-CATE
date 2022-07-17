import numba
import numpy as np
from tnw_cate_model import train_kernel, train_alpha, DynamicModel

def calc_mse(y1, y2):
    return np.mean((np.ravel(y1) - np.ravel(y2)) ** 2)

@numba.njit
def make_spec_set(x_in, x_p, y_in, y_p, n, m, mlp_coef):
    idx = np.arange(x_in.shape[0])
    x_in_out = np.zeros((x_p.shape[0], mlp_coef, n, m))
    y_in_out = np.zeros((x_p.shape[0], mlp_coef, n))
    x_p_out = np.zeros((x_p.shape[0], mlp_coef, m))
    labels = np.zeros((x_p.shape[0], mlp_coef))
    for i in range(x_p.shape[0]):
        for j in range(mlp_coef):
            x_p_out[i, j, :] = x_p[i]    
        # x_p_out[i, ...] = np.repeat(x_p[np.newaxis, i], mlp_coef, axis=0)
        labels[i, :] = y_p[i]
        for j in range(mlp_coef):
            cur_idx = np.random.choice(idx, n, False)
            x_in_out[i, j, ...] = x_in[cur_idx]
            y_in_out[i, j, :] = y_in[cur_idx]
    x_in_out = np.reshape(x_in_out, (-1, n, m))
    y_in_out = np.reshape(y_in_out, (-1, n))
    x_p_out = np.reshape(x_p_out, (-1, m))
    labels = np.reshape(labels, (-1, 1))
    return x_in_out, y_in_out, x_p_out, labels

@numba.njit
def make_part_set(x_in, y_in, x_p, y_p, n, m, tasks):
    idx = np.arange(x_in.shape[0])
    x_in_res = np.zeros((tasks, n, m))
    y_in_res = np.zeros((tasks, n))
    x_p_res = np.zeros((tasks, m))
    labels = np.zeros((tasks))
    cur_task = 0
    mlp_coef = tasks // x_p.shape[0]
    for i in range(x_p.shape[0] - 1):
        for j in range(mlp_coef):
            x_p_res[i * mlp_coef + j, ...] = x_p[i]
            labels[i * mlp_coef + j] = y_p[i]
        # x_p_res[i * mlp_coef : (i + 1) * mlp_coef, ...] = np.repeat(x_p[np.newaxis, i], mlp_coef, axis=0)
        # labels[i * mlp_coef : (i + 1) * mlp_coef] = y_p[i]
        for j in range(mlp_coef):
            cur_idx = np.random.choice(idx, n, False)
            x_in_res[i * mlp_coef + j, ...] = x_in[cur_idx]
            y_in_res[i * mlp_coef + j, :] = y_in[cur_idx]
            cur_task += 1
        idx[i] = i
    remain = tasks - cur_task
    for i in range(remain):
        x_p_res[cur_task + i, ...] = x_p[-1]
        labels[cur_task + i] = y_p[-1]
    # x_p_res[cur_task:, ...] = np.repeat(x_p[np.newaxis, -1], remain, axis=0)
    # labels[cur_task:] = y_p[-1]
    for j in range(remain):
        cur_idx = np.random.choice(idx, n, False)
        x_in_res[cur_task + j, ...] = x_in[cur_idx]
        y_in_res[cur_task + j, :] = y_in[cur_idx]
    return x_in_res, y_in_res, x_p_res, labels

class dataset_getter():
    def get_train_set(self):
        return self.train_x, self.train_y, self.train_w

    def get_cotrol_treat_sets(self):
        return self.control_x, self.control_y, self.treat_x, self.treat_y

    def get_val_set(self):
        return self.val_set_c, self.val_labels_c, self.val_set_t, self.val_labels_t

    def get_test_set(self):
        return self.test_x, self.test_control, self.test_treat, self.test_cate

class dataset_param_getter(dataset_getter):
    def __init__(self, control_func, treatment_func, t_bounds):
        self.control_func = control_func
        self.treatment_func = treatment_func
        self.t_bounds = t_bounds

    def make_set(self, control_size, treat_part, test_size):
        treat_size = int(treat_part * control_size)
        val_size_c = int(control_size * 0.2)
        val_size_t = int(treat_size * 0.5)
        train_t = np.random.uniform(self.t_bounds[0], self.t_bounds[1], control_size + treat_size)
        train_w = np.zeros(control_size + treat_size)
        w_idx = np.random.choice(range(train_w.shape[0]), treat_size, replace=False)
        train_w[w_idx] = 1
        train_x = self.control_func.calc_x(train_t)
        control_x = train_x[train_w == 0]
        treat_x = train_x[train_w == 1]
        control_y = self.control_func.calc_y(train_t[train_w == 0])
        treat_y = self.treatment_func.calc_y(train_t[train_w == 1])
        train_y = np.empty(control_size + treat_size)
        train_y[train_w == 0] = control_y
        train_y[train_w == 1] = treat_y
        mean = np.mean(train_y)
        std = np.std(train_y)
        control_y = (control_y - mean) / std
        treat_y = (treat_y - mean) / std
        train_y = (train_y - mean) / std

        self.train_x, self.train_y, self.train_w = train_x, train_y, train_w
        self.control_x, self.control_y, self.treat_x, self.treat_y = control_x, control_y, treat_x, treat_y


        val_t_c = np.random.uniform(self.t_bounds[0], self.t_bounds[1], val_size_c)
        self.val_set_c = self.control_func.calc_x(val_t_c) 
        self.val_labels_c = (self.control_func.calc_y(val_t_c) - mean) / std
        # actually there is no validation set for treatment
        self.val_set_t = np.zeros((val_size_t, self.control_func.m))
        self.val_labels_t = np.zeros(val_size_t)

        test_t = np.random.uniform(self.t_bounds[0], self.t_bounds[1], test_size)
        self.test_x = self.control_func.calc_x(test_t)
        self.test_control = (self.control_func.calc_y(test_t) - mean) / std
        self.test_treat = (self.treatment_func.calc_y(test_t) - mean) / std
        self.test_cate = self.test_treat - self.test_control

        self.val_size_c = val_size_c

    


class dataset_sa_func_getter(dataset_getter):
    def __init__(self, func, x_bounds):
        self.func = func
        self.x_bounds = x_bounds

    def make_set(self, control_size, treat_part, test_size):
        treat_size = int(treat_part * control_size)
        val_size_c = int(control_size * 0.2)
        val_size_t = int(treat_size * 0.5)
        train_x = np.random.uniform(self.x_bounds[0], self.x_bounds[1], (control_size + treat_size, self.func.m))
        train_w = np.zeros(control_size + treat_size)
        w_idx = np.random.choice(range(train_w.shape[0]), treat_size, replace=False)
        train_w[w_idx] = 1
        control_x = train_x[train_w == 0]
        treat_x = train_x[train_w == 1]
        control_y = self.func.get_control_y(control_x)
        treat_y = self.func.get_treat_y(treat_x)
        train_y = np.empty(control_size + treat_size)
        train_y[train_w == 0] = control_y
        train_y[train_w == 1] = treat_y
        mean = np.mean(train_y)
        std = np.std(train_y)
        control_y = (control_y - mean) / std
        treat_y = (treat_y - mean) / std
        train_y = (train_y - mean) / std

        self.train_x, self.train_y, self.train_w = train_x, train_y, train_w
        self.control_x, self.control_y, self.treat_x, self.treat_y = control_x, control_y, treat_x, treat_y

        self.val_set_c = np.random.uniform(self.x_bounds[0], self.x_bounds[1], (val_size_c, self.func.m))
        self.val_labels_c = (self.func.get_control_y(self.val_set_c) - mean) / std
        # actually there is no validation set for treatment
        self.val_set_t = np.zeros((val_size_t, self.func.m))
        self.val_labels_t = np.zeros(val_size_t)

        self.test_x = np.random.uniform(self.x_bounds[0], self.x_bounds[1], (test_size, self.func.m))
        self.test_control = (self.func.get_control_y(self.test_x) - mean) / std
        self.test_treat = (self.func.get_treat_y(self.test_x) - mean) / std
        self.test_cate = self.test_treat - self.test_control

        self.val_size_c = val_size_c

def get_basic_dynamic_model(data_getter, m, n, epochs_num, mlp_coef, mlp_coef_val, learning_rate, seed, batch_size, patience):
    *val_set_kernel_0, val_labels_kernel_0 = make_spec_set(data_getter.control_x, data_getter.val_set_c, data_getter.control_y, data_getter.val_labels_c, n, m, mlp_coef_val)
    model_static, _ = train_kernel(data_getter.control_x, data_getter.control_y, n, m, mlp_coef, epochs_num, (val_set_kernel_0, val_labels_kernel_0), seed, batch_size, patience, learning_rate)
    return DynamicModel(model_static, False)

def get_alpha_dynamic_model(data_getter, m, n_c, n_t, epochs_num, mlp_coef_val, learning_rate, seed, batch_size, tasks, alpha, patience):
    *val_set_alpha_t, val_labels_alpha_t = make_part_set(data_getter.treat_x, data_getter.treat_y, data_getter.val_set_t, data_getter.val_labels_t, n_t, m, mlp_coef_val * data_getter.val_size_c) 
    *val_set_alpha_c, val_labels_alpha_c = make_part_set(data_getter.control_x, data_getter.control_y,  data_getter.val_set_c, data_getter.val_labels_c, n_c, m, mlp_coef_val * data_getter.val_size_c)
    val_set_alpha = list(val_set_alpha_c) + list(val_set_alpha_t)
    val_labels_alpha = [val_labels_alpha_c, val_labels_alpha_t]
    alpha_model_static, _ = train_alpha(data_getter.control_x, data_getter.control_y, data_getter.treat_x, data_getter.treat_y, n_c, n_t, m, epochs_num, tasks, (val_set_alpha, val_labels_alpha), seed, alpha, batch_size, patience, learning_rate)
    return DynamicModel(alpha_model_static, False)
