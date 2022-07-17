import numba
import numpy as np
import tensorflow as tf
from keras import Input
from keras import Model
from keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

class DynamicModel():
    def __init__(self, static_model, f_gpu=True, f_w_feat=False) -> None:
        self.kernel = static_model.get_layer('kernel')
        self.w_feat = f_w_feat
        if f_gpu:
            self.device = '/device:gpu:0'
            self.batch = 2048
        else:
            self.device = '/device:cpu:0'
            self.batch = 256
    
    def predict(self, dataset):
        x_i = dataset[0]
        y_i = dataset[1]
        x_p = dataset[2]
        if self.w_feat:
            w_i = x_i[..., -1]
            w_p = x_p[:, -1]
            x_i = x_i[..., :-1]
            x_p = x_p[:, :-1]
        mean = np.mean(x_i, axis=1, keepdims=True)
        sigma = np.std(x_i, axis=1, keepdims=True)
        x_i = (x_i - mean) / sigma
        x_p = (x_p - mean[:, 0, :]) / sigma[:, 0, :]
        if self.w_feat:
            x_i = np.concatenate((x_i, w_i[..., np.newaxis]), axis=-1)
            x_p = np.concatenate((x_p, w_p[:, np.newaxis]), axis=-1)
        n = x_i.shape[1]
        K = np.zeros((x_i.shape[0], n))
        with tf.device(self.device):
            for i in range(n):
                K[:, i] = np.ravel(self.kernel.predict((x_i[:, i, :], x_p), batch_size=self.batch))
        sum = np.sum(K, axis=-1, keepdims=True)
        dot = np.sum(y_i * K, axis=1, keepdims=True)
        return dot / sum

class Normalization_layer(layers.Layer):
    def __init__(self, trainable=False, name='Normalization', dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
    
    def call(self, input):
        x_i = input[0] # (None, n, m)
        x_p = input[1] # (None, m)
        mean = tf.reduce_mean(x_i, axis=1, keepdims=True) # (None, 1, m)
        sigma = tf.math.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_i, mean)), axis=1, keepdims=True))
        res_i = tf.math.divide_no_nan(tf.subtract(x_i, mean), sigma)
        mean = mean[:, 0, :]
        sigma = sigma[:, 0, :]
        res_p = tf.math.divide_no_nan(tf.subtract(x_p, mean), sigma)
        return res_i, res_p

def get_d_model(m, seed=None): # dimension of 1 point
    initializer = tf.keras.initializers.GlorotNormal(seed=seed)

    s = int(np.sqrt(m))

    x_i = Input(shape=(m), name='x_i')
    x_p = Input(shape=(m), name='x^')
    

    l_sparse = layers.Dense(2 * m, activation='relu', kernel_initializer=initializer) # 2 * m
    x_i_sparse = l_sparse(x_i)
    x_p_sparse = l_sparse(x_p)
    total_input = tf.math.abs(x_i_sparse - x_p_sparse)
    
    topology = [
        {'units': 2 * s, 'activation': 'tanh', 'kernel_initializer': initializer},
        {'units': s, 'activation': 'tanh', 'kernel_initializer': initializer},
        {'units': 1, 'activation': 'softplus', 'kernel_initializer': initializer}
    ]

    layer = total_input
    for info in topology:
        layer = layers.Dense(**info)(layer)

    return Model([x_i, x_p], layer, name='kernel')

def get_full_model(n, m, seed):
    x_input = Input(shape=(n, m), name='x_i') 
    y_input = Input(shape=(n), name='y_i')
    x_p_input = Input(shape=(m), name='x_predict')

    x, x_p = Normalization_layer()((x_input, x_p_input))

    kernel = get_d_model(m, seed)
    d_single_res = []
    for i in range(n):
        d_single_res.append(kernel([x[:, i, :], x_p]))
    d_output = layers.concatenate(d_single_res, axis=-1)
    dot = layers.Dot(axes=1)([d_output, y_input])
    sum = tf.math.reduce_sum(d_output, axis=1, keepdims=True)

    return Model([x_input, y_input, x_p_input], dot / sum)

def get_alpha_model(n_c, n_t, m, seed=None):
    x_i_c = Input(shape=(n_c, m), name='x_i_c')
    y_i_c = Input(shape=(n_c), name='y_i_c')
    x_p_c = Input(shape=(m), name='x_predict_c')
    x_i_t = Input(shape=(n_t, m), name='x_i_t')
    y_i_t = Input(shape=(n_t), name='y_i_t')
    x_p_t = Input(shape=(m), name='x_predict_t')

    norm_layer = Normalization_layer()

    x_i_c_norm, x_p_c_norm = norm_layer((x_i_c, x_p_c))
    x_i_t_norm, x_p_t_norm = norm_layer((x_i_t, x_p_t))

    kernel = get_d_model(m, seed)
    d_control = []
    for i in range(n_c):
        d_control.append(kernel([x_i_c_norm[:, i, :], x_p_c_norm]))
    d_treat = []
    for i in range(n_t):
        d_treat.append(kernel([x_i_t_norm[:, i, :], x_p_t_norm]))
    cnt_output = layers.concatenate(d_control, axis=-1)
    trt_output = layers.concatenate(d_treat, axis=-1)
    cnt_dot = layers.Dot(axes=1)([cnt_output, y_i_c])
    trt_dot = layers.Dot(axes=1)([trt_output, y_i_t])
    cnt_sum = tf.math.reduce_sum(cnt_output, axis=1, keepdims=True)
    trt_sum = tf.math.reduce_sum(trt_output, axis=1, keepdims=True)
    cnt_res = tf.math.divide_no_nan(cnt_dot, cnt_sum)
    trt_res = tf.math.divide_no_nan(trt_dot, trt_sum)
    return Model([x_i_c, y_i_c, x_p_c, x_i_t, y_i_t, x_p_t], [cnt_res, trt_res])

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

def train_kernel(X_train, y_train, n, m, mlp_coef, epochs_num, validation, seed, batch_size, patience=3, learning_rate=0.01):
    model = get_full_model(n, m, seed)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience = patience, monitor = "val_loss",  mode = "min", restore_best_weights = True)
    adam = Adam(learning_rate=learning_rate) #0.0005; 0.01
    model.compile(optimizer=adam, loss='mean_squared_error')

    train_gen = TrainGenerator(X_train, y_train, n, m, mlp_coef, batch_size)
    hist = model.fit(train_gen, batch_size=batch_size, epochs=epochs_num, callbacks=[early_stopping], verbose=2, validation_data=validation).history
    return model, len(hist['loss'])

def train_alpha(cnt_x, cnt_y, trt_x, trt_y, n_c, n_t, m, epochs_num, tasks_num, validation, seed, alpha, batch_size, patience=5, learning_rate=0.002):
    model = get_alpha_model(n_c, n_t, m, seed)
    output_name = model.output_names[0]
    early_stopping = tf.keras.callbacks.EarlyStopping(patience = patience, monitor = f"val_{output_name}_loss",  mode = "min", restore_best_weights = True)
    adam = Adam(learning_rate=learning_rate) #0.0005
    model.compile(optimizer=adam, loss=['mse', 'mse'], loss_weights=[1.0, alpha])
    train_gen = AlphaGenerator(cnt_x, cnt_y, trt_x, trt_y, n_c, n_t, m, tasks_num, batch_size)
    hist = model.fit(train_gen, batch_size=batch_size, epochs=epochs_num, callbacks=[early_stopping], verbose=2, validation_data=validation).history
    return model, len(hist['loss'])