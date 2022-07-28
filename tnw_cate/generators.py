import numba
import numpy as np
from tensorflow.keras.utils import Sequence

@numba.njit
def alpha_tasks_gen(x, y, n, m, tasks_num):
    idx = np.arange(1, x.shape[0])
    x_in = np.zeros((tasks_num, n, m))
    y_in = np.zeros((tasks_num, n))
    x_p = np.zeros((tasks_num, m))
    labels = np.zeros((tasks_num))
    cur_task = 0
    mlp_coef = tasks_num // x.shape[0]
    for i in range(x.shape[0] - 1):
        for j in range(mlp_coef):
            x_p[i * mlp_coef  + j, ...] = x[i]
            labels[i * mlp_coef + j] = y[i]
        for j in range(mlp_coef):
            cur_idx = np.random.choice(idx, n, False)
            x_in[i * mlp_coef + j, ...] = x[cur_idx]
            y_in[i * mlp_coef + j, :] = y[cur_idx]
            cur_task += 1
        idx[i] = i
    remain = tasks_num - cur_task
    for i in range(remain):
        x_p[cur_task + i, ...] = x[-1]
        labels[cur_task + i] = y[-1]
    for j in range(remain):
        cur_idx = np.random.choice(idx, n, False)
        x_in[cur_task + j, ...] = x[cur_idx]
        y_in[cur_task + j, :] = y[cur_idx]
    return x_in, y_in, x_p, labels

class AlphaGenerator(Sequence):
    def __init__(self, cnt_x, cnt_y, trt_x, trt_y, n_c, n_t, m, tasks_num, batch_size):
        self.cnt_x = cnt_x
        self.cnt_y = cnt_y
        self.trt_x = trt_x
        self.trt_y = trt_y
        self.n_c = n_c
        self.n_t = n_t
        self.tasks_num = tasks_num
        self.batch_size = batch_size
        self.m = m
        self.on_epoch_end()

    def __len__(self):
        return np.int0(np.ceil(self.tasks_num / self.batch_size))

    def __getitem__(self, idx):
        data_cnt = [ar[idx * self.batch_size:(idx + 1) * self.batch_size] for ar in self.cnt_data]
        labels_cnt = self.cnt_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        data_trt = [ar[idx * self.batch_size:(idx + 1) * self.batch_size] for ar in self.trt_data]
        labels_trt = self.trt_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return data_cnt + data_trt, [labels_cnt, labels_trt]
    
    def on_epoch_end(self):
        *con_data, self.cnt_labels = alpha_tasks_gen(self.cnt_x, self.cnt_y, self.n_c, self.m, self.tasks_num)
        *treat_data, self.trt_labels = alpha_tasks_gen(self.trt_x, self.trt_y, self.n_t, self.m, self.tasks_num)
        self.cnt_data = list(con_data)
        self.trt_data = list(treat_data)

@numba.njit
def make_train_set(x, y, n, m, mlp_coef):
    idx = np.asarray(list(range(1, x.shape[0])))
    x_in = np.zeros((x.shape[0], mlp_coef, n, m))
    y_in = np.zeros((x.shape[0], mlp_coef, n))
    x_p = np.zeros((x.shape[0], mlp_coef, m))
    labels = np.zeros((x.shape[0], mlp_coef))
    for i in range(x.shape[0]):
        # x_r = np.reshape(x[i], (1, m))
        # check_2 = np.repeat(x_r, mlp_coef, np.int64(0))
        # x_p[i, ...] = np.repeat(x_r, mlp_coef, np.int64(0))
        for j in range(mlp_coef):
            x_p[i, j, :] = x[i]
        labels[i, :] = y[i]
        for j in range(mlp_coef):
            cur_idx = np.random.choice(idx, n, False)
            x_in[i, j, ...] = x[cur_idx]
            y_in[i, j, :] = y[cur_idx]
        if i < len(idx):
            idx[i] = i
    x_in = np.reshape(x_in, (-1, n, m))
    y_in = np.reshape(y_in, (-1, n))
    x_p = np.reshape(x_p, (-1, m))
    labels = np.reshape(labels, (-1, 1))
    return x_in, y_in, x_p, labels

class TrainGenerator(Sequence):
    def __init__(self, control_x, control_y, n, m, mlp_coef, batch_size) -> None:
        self.control_x = control_x
        self.control_y = control_y
        self.mlp_coef = mlp_coef
        self.batch_size = batch_size
        self.n = n
        self.m = m
        self.on_epoch_end()

    def __len__(self):
        return np.int0(np.ceil(self.control_x.shape[0] * self.mlp_coef / self.batch_size))

    def __getitem__(self, idx):
        return [ar[idx * self.batch_size:(idx + 1) * self.batch_size] for ar in self.data], self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
                 
    def on_epoch_end(self):
        x, y, x_p, self.labels = make_train_set(self.control_x, self.control_y, self.n, self.m, self.mlp_coef)
        self.data = [x, y, x_p]