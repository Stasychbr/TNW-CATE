import numpy as np
import tensorflow as tf
from keras import Input
from keras import Model
from keras import layers


class DynamicModel():
    def __init__(self, static_model, f_gpu=True, f_w_feat=False, f_normalize=True) -> None:
        self.kernel = static_model.get_layer('kernel')
        self.w_feat = f_w_feat
        self.f_norm = f_normalize
        if f_gpu:
            self.device = '/device:gpu:0'
            self.batch = 2048
        else:
            self.device = '/device:cpu:0'
            self.batch = 8192

    def predict(self, dataset):
        x_i = dataset[0]
        y_i = dataset[1]
        x_p = dataset[2]
        if self.w_feat:
            w_i = x_i[..., -1]
            w_p = x_p[:, -1]
            x_i = x_i[..., :-1]
            x_p = x_p[:, :-1]
        if self.f_norm:
            mean = np.mean(x_i, axis=1, keepdims=True)
            sigma = np.std(x_i, axis=1, keepdims=True)
            x_i = (x_i - mean) / sigma
            x_p = (x_p - mean[:, 0, :]) / sigma[:, 0, :]
        if self.w_feat:
            x_i = np.concatenate((x_i, w_i[..., np.newaxis]), axis=-1)
            x_p = np.concatenate((x_p, w_p[:, np.newaxis]), axis=-1)
        n = x_i.shape[1]
        K = np.zeros((x_i.shape[0], n))
        x_p_repeat = np.repeat(x_p, n, 0)
        with tf.device(self.device):
            K = self.kernel.predict(
                (x_i.reshape((-1, x_i.shape[-1])), x_p_repeat), batch_size=self.batch, verbose=0).reshape((-1, n))
        sum = np.sum(K, axis=-1, keepdims=True)
        dot = np.sum(y_i * K, axis=1, keepdims=True)
        return dot / sum


class Normalization_layer(layers.Layer):
    def __init__(self, trainable=False, name='Normalization', dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def call(self, input):
        x_i = input[0]  # (None, n, m)
        x_p = input[1]  # (None, m)
        mean = tf.reduce_mean(x_i, axis=1, keepdims=True)  # (None, 1, m)
        sigma = tf.math.sqrt(tf.reduce_mean(
            tf.square(tf.subtract(x_i, mean)), axis=1, keepdims=True))
        res_i = tf.math.divide_no_nan(tf.subtract(x_i, mean), sigma)
        mean = mean[:, 0, :]
        sigma = sigma[:, 0, :]
        res_p = tf.math.divide_no_nan(tf.subtract(x_p, mean), sigma)
        return res_i, res_p


def get_d_model(m, topology=None, seed=None):  # dimension of 1 point
    initializer = tf.keras.initializers.GlorotNormal(seed=seed)

    s = int(np.sqrt(m))

    x_i = Input(shape=(m), name='x_i')
    x_p = Input(shape=(m), name='x^')

    l_sparse = layers.Dense(2 * m, activation='relu', kernel_initializer=initializer)  # 2 * m
    x_i_sparse = l_sparse(x_i)
    x_p_sparse = l_sparse(x_p)
    total_input = tf.math.abs(x_i_sparse - x_p_sparse)

    if topology is None:
        topology = [
            {'units': 2 * s, 'activation': 'tanh', 'kernel_initializer': initializer},
            {'units': s, 'activation': 'tanh', 'kernel_initializer': initializer},
            {'units': 1, 'activation': 'softplus', 'kernel_initializer': initializer}
        ]

    layer = total_input
    for info in topology:
        layer = layers.Dense(**info)(layer)

    return Model([x_i, x_p], layer, name='kernel')


def get_full_model(n, m, topology=None, seed=None, f_normalize=True):
    x_input = Input(shape=(n, m), name='x_i')
    y_input = Input(shape=(n), name='y_i')
    x_p_input = Input(shape=(m), name='x_predict')

    if f_normalize:
        x, x_p = Normalization_layer()((x_input, x_p_input))
    else:
        x, x_p = x_input, x_p_input

    kernel = get_d_model(m, topology, seed)
    x_p_repeat = tf.repeat(x_p, n, 0)
    d_output = tf.reshape(kernel((tf.reshape(x, (-1, m)), x_p_repeat)), (-1, n))
    dot = layers.Dot(axes=1)([d_output, y_input])
    sum = tf.math.reduce_sum(d_output, axis=1, keepdims=True)

    return Model([x_input, y_input, x_p_input], dot / sum)


def get_alpha_model(n_c, n_t, m, topology=None, seed=None, f_normalize=True):
    x_i_c = Input(shape=(n_c, m), name='x_i_c')
    y_i_c = Input(shape=(n_c), name='y_i_c')
    x_p_c = Input(shape=(m), name='x_predict_c')
    x_i_t = Input(shape=(n_t, m), name='x_i_t')
    y_i_t = Input(shape=(n_t), name='y_i_t')
    x_p_t = Input(shape=(m), name='x_predict_t')

    if f_normalize:
        norm_layer = Normalization_layer()

        x_i_c_norm, x_p_c_norm = norm_layer((x_i_c, x_p_c))
        x_i_t_norm, x_p_t_norm = norm_layer((x_i_t, x_p_t))
    else:
        x_i_c_norm, x_p_c_norm = x_i_c, x_p_c
        x_i_t_norm, x_p_t_norm = x_i_t, x_p_t

    kernel = get_d_model(m, topology, seed)
    x_p_c_repeat = tf.repeat(x_p_c_norm, n_c, 0)
    cnt_output = tf.reshape(kernel((tf.reshape(x_i_c_norm, (-1, m)), x_p_c_repeat)), (-1, n_c))

    x_p_t_repeat = tf.repeat(x_p_t_norm, n_t, 0)
    trt_output = tf.reshape(kernel((tf.reshape(x_i_t_norm, (-1, m)), x_p_t_repeat)), (-1, n_t))
    cnt_dot = layers.Dot(axes=1)([cnt_output, y_i_c])
    trt_dot = layers.Dot(axes=1)([trt_output, y_i_t])
    cnt_sum = tf.math.reduce_sum(cnt_output, axis=1, keepdims=True)
    trt_sum = tf.math.reduce_sum(trt_output, axis=1, keepdims=True)
    cnt_res = tf.math.divide_no_nan(cnt_dot, cnt_sum)
    trt_res = tf.math.divide_no_nan(trt_dot, trt_sum)
    return Model([x_i_c, y_i_c, x_p_c, x_i_t, y_i_t, x_p_t], [cnt_res, trt_res])
