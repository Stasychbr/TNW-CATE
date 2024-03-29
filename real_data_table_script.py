import pickle
from time import time
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from treatment_frameworks import *
from utility import *
from other_models import *
from funcs import get_ihdp_setup


if __name__ == '__main__':

    cur_setup = get_ihdp_setup
    val_part = 0.2
    test_part = 0.4
    n_splits = int(round((1 - test_part) / val_part, 0))

    alpha = 0.1
    batch_size = 256
    learning_rate_a = 0.005
    learning_rate = 0.01
    epochs_num_alpha = 50
    epochs_num_ordinary = 100
    patience = 7
    tasks_alpha = 100000
    iters = 5

    tasks_num = 200000
    n = 50
    mlp_coef_val = 300

    seed = int(time()) % 2048

    ser_name = 'ihdp table'

    nw_cv_grid = {
        'gamma': [10 ** i for i in range(-8, 11)] + [0.5, 5, 50, 100, 200, 500, 700],
    }
    forest_cv_grid = {
        'n_estimators': [10, 50, 100, 300],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_samples_leaf': [1, 0.05, 0.1, 0.2],
    }

    results = {
        'T-NW control mse': [],
        'T-NW treat mse': [],
        'T-NW CATE mse': [],
        'S-NW control mse': [],
        'S-NW treat mse': [],
        'S-NW CATE mse': [],
        'X-NW CATE mse': [],
        'T-Learner control mse': [],
        'T-Learner treat mse': [],
        'T-Learner CATE mse': [],
        'S-Learner control mse': [],
        'S-Learner treat mse': [],
        'S-Learner CATE mse': [],
        'X-Learner CATE mse': [],
        'Kernel control': [],
        'Kernel treat': [],
        'Kernel CATE': [],
        'time': [],
    }

    tf.config.set_visible_devices([], 'GPU')
    np.seterr(invalid='ignore')
    np.random.seed(seed)

    try:
        for i in range(iters):
            seed = np.random.randint(0, 2048)
            start_time = time()
            setup = cur_setup(seed)
            setup.make_set(val_part, test_part)
            train_x, train_y, train_w = setup.get_train_set()
            control_x, control_y, treat_x, treat_y = setup.get_cotrol_treat_sets()
            test_x, test_control, test_treat,  test_cate = setup.get_test_set()
            val_set_c, val_labels_c, val_set_t, val_labels_t = setup.get_val_set()

            m = train_x.shape[1]
            control_size = np.count_nonzero(train_w == 0)
            treat_size = np.count_nonzero(train_w == 1)
            mlp_coef = tasks_num // control_size
            model = get_basic_dynamic_model(setup, m, n, epochs_num_ordinary, mlp_coef, 
                                            mlp_coef_val, learning_rate, seed, batch_size, patience, f_normalize=False)

            *test_data_control, cnt_label = make_spec_set(control_x,
                                                          test_x, control_y, test_control, control_size, m, 1)
            *test_data_treat, trt_label = make_spec_set(treat_x,
                                                        test_x, treat_y, test_treat, treat_size, m, 1)
            cnt_pred = model.predict(test_data_control)
            results['Kernel control'].append(calc_mse(cnt_pred, cnt_label))
            n_t = treat_size // 2
            n_c = n_t
            alpha_model = get_alpha_dynamic_model(setup, m, n_c, n_t, epochs_num_alpha, mlp_coef_val,
                                                  learning_rate, seed, batch_size, tasks_alpha, alpha, patience, f_normalize=False)
            trt_pred_a = alpha_model.predict(test_data_treat)
            trt_pred_a = model.predict(test_data_treat)
            results['Kernel treat'].append(calc_mse(trt_pred_a, trt_label))
            results['Kernel CATE'].append(calc_mse(trt_pred_a - cnt_pred, trt_label - cnt_label))

            other_models = {
                'T-Learner': (make_t_learner, (val_set_c, val_labels_c, val_set_t, val_labels_t, RandomForestRegressor, forest_cv_grid), (train_x, train_y, train_w)),
                'S-Learner': (make_s_learner, (val_set_c, val_labels_c, val_set_t, val_labels_t, RandomForestRegressor, forest_cv_grid), (train_x, train_y, train_w)),
                'X-Learner': (make_x_learner, (val_set_c, val_labels_c, val_set_t, val_labels_t, RandomForestRegressor, forest_cv_grid), (train_x, train_y, train_w)),
                'T-NW': (make_t_learner, (val_set_c, val_labels_c, val_set_t, val_labels_t, KernelRegression, nw_cv_grid), (train_x, train_y, train_w)),
                'S-NW': (make_s_learner, (val_set_c, val_labels_c, val_set_t, val_labels_t, KernelRegression, nw_cv_grid), (train_x, train_y, train_w)),
                'X-NW': (make_x_learner, (val_set_c, val_labels_c, val_set_t, val_labels_t, KernelRegression, nw_cv_grid), (train_x, train_y, train_w))
            }

            for key, val in other_models.items():
                instance = val[0](*val[1], n_splits, random_state=seed)
                instance.fit(*val[2])
                if (hasattr(instance, 'predict_control')):
                    control_pred = instance.predict_control(test_x)
                    results[f'{key} control mse'].append(calc_mse(control_pred, test_control))
                if (hasattr(instance, 'predict_treat')):
                    treat_pred = instance.predict_treat(test_x)
                    results[f'{key} treat mse'].append(calc_mse(treat_pred, test_treat))
                cate_pred = instance.predict(test_x)
                results[f'{key} CATE mse'].append(calc_mse(cate_pred, test_cate))

            print('itetation ', i + 1, '/', iters)
            cur_time = time() - start_time
            print('time: ', cur_time)
            results['time'].append(cur_time)

    except KeyboardInterrupt:
        pass
    # except Exception as e:
    #     print(e)
    #     pass
    print('time elapsed: ', round(np.sum(results['time']), 0), ' s.')

    print('result:')
    for key, val in results.items():
        if key != 'time':
            print(key, ': ', round(np.mean(val), 6))

    params = {
        'batch_size': batch_size,
        'epochs_num_alpha': epochs_num_alpha,
        'epochs_num_ordinary': epochs_num_ordinary,
        'mlp_coef': mlp_coef,
        'patience': patience,
        'm': m,
        'n': n,
        'n_c': n_c,
        'n_t': n_t,
        'alpha': alpha,
        'tasks': tasks_alpha,
        'iters': iters,
        'control_size': control_size,
        'treat_part': 1 - test_part - val_part,
        'treat_size': treat_size,
        'test_size': test_x.shape[0],
        'seed': seed,
        'learning_rate': learning_rate,
        'val_part_c': 0.2,
        'val_part_t': 0,
        'mlp_coef_val': mlp_coef_val
    }

    results['params'] = params
    with open(f'res_dicts/{ser_name}.pk', 'wb') as file:
        pickle.dump(results, file)
