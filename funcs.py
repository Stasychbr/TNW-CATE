import numpy as np
from utility import dataset_param_getter, dataset_sa_func_getter

class pow_func():
    def __init__(self, m, scale, sigma, mean) -> None:
        self.m = m
        self.scale = scale
        self.sigma = sigma
        self.mean = mean
    
    def calc_x(self, t):
        a = 1 / np.sqrt(self.m)
        pows = [a * (i + 1) for i in range(self.m)]
        X = np.empty((t.shape[0], self.m))
        for i, p in enumerate(pows):
            if 0.8 < p < 1.6:
                X[:, i] = np.random.normal(0, 1, t.shape[0])
            else:
                X[:, i] = np.power(t, a * (i + 1))
        return X

    def calc_y(self, t):
        return self.scale * np.exp(-(t - self.mean) ** 2 / self.sigma)


class log_func():
    def __init__(self, m, coeffs_neg, coeffs_pos, y_pos = True) -> None:
        self.log_coeffs = np.empty(m)
        self.m = m
        neg_num = m // 2
        self.log_coeffs[:neg_num] = np.random.uniform(coeffs_neg[0], coeffs_neg[1], neg_num)
        pos_num = m - neg_num
        self.log_coeffs[neg_num:neg_num + pos_num] = np.random.uniform(coeffs_pos[0], coeffs_pos[1], pos_num)
        rng = np.random.default_rng()
        rng.shuffle(self.log_coeffs)
        if y_pos:
            self.func_coeffs = np.random.uniform(coeffs_pos[0], coeffs_pos[1], 1)
        else:
            self.func_coeffs = np.random.uniform(coeffs_neg[0], coeffs_neg[1], 1)

    def calc_x(self, t):
        res = np.empty((t.shape[0], self.m))
        for i in range(self.m):
            res[:, i] = self.log_coeffs[i] * np.log(t)
        return res

    def calc_y(self, t):
        return (np.log(t) + np.sin(t)) * self.func_coeffs[0]


class Spiral_linear():
    def __init__(self, coeffs_bounds, m):
        self.a = np.random.uniform(coeffs_bounds[0], coeffs_bounds[1])
        self.bias = np.random.uniform(coeffs_bounds[0], coeffs_bounds[1])
        self.m = m
    
    def calc_x(self, t):
        X = np.empty((t.shape[0], self.m), dtype=np.float32)
        for i in range(self.m):
            if i % 2 == 0:
                X[:, i] = t * np.sin((i // 2 + 1) * t)
            else:
                X[:, i] = t * np.cos((i // 2 + 1) * t)
        return X
    
    def calc_y(self, t):
        Y = self.a * t + self.bias
        return Y

class simple_cate():
    def __init__(self, m):
        self.coeffs = np.random.uniform(-5, 5, (1, m))
        self.m = m
    
    def get_control_y(self, x):
        linear_y = np.sum(self.coeffs * x, axis=-1)
        mask = x[:, 0] > 0.5
        return linear_y + 5 * mask
    
    def get_treat_y(self, x):
        base = self.get_control_y(x)
        mask = x[:, 1] > 0.1
        return base + 8 * mask

def get_log_setup(m):
    t_bounds = (0.5, 5)
    treat_func = log_func(m, (-4, -1), (1, 4), False)
    control_func = log_func(m, (-4, -1), (1, 4), True)
    return dataset_param_getter(control_func, treat_func, t_bounds)

def get_pow_setup(m):
    t_bounds = (0, 5)
    treat_scale = np.random.uniform(2, 4)
    treat_sigma = np.random.uniform(1, 2)
    treat_func = pow_func(m, treat_scale, treat_sigma, (t_bounds[1] - t_bounds[0]) / 2)
    control_scale = np.random.uniform(1, 2)
    control_sigma = np.random.uniform(0.25, 1)
    control_func = pow_func(m, control_scale, control_sigma, (t_bounds[1] - t_bounds[0]) / 2)
    return dataset_param_getter(control_func, treat_func, t_bounds)

def get_spiral_setup(m):
    t_bounds = (0, 10)
    treat_func = Spiral_linear((8, 10), m)
    control_func = Spiral_linear((1, 4), m)
    return dataset_param_getter(control_func, treat_func, t_bounds)

def get_indic_setup(m):
    x_bounds = (-1, 1)
    func = simple_cate(m)
    return dataset_sa_func_getter(func, x_bounds)
