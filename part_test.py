import pickle
from time import time
import numpy as np
from treatment_frameworks import *
from utility import *
from other_models import *
from funcs import get_log_setup, get_pow_setup, get_spiral_setup, get_indic_setup
import tensorflow as tf

if __name__ == '__main__':

    cur_setup = get_indic_setup

    alpha = 0.5
    batch_size = 256
    learning_rate_a = 0.002
    learning_rate = 0.01
    epochs_num_alpha = 100
    epochs_num_ordinary = 30
    patience = 5
    m = 10
    n = 100
    n_c = 10
    n_t = 10
    tasks = 200000
    iters = 1
    n_splits = 3

    control_size = 200
    treat_parts = [i / 10 for i in range(1, 6)]
    test_size = 1000

    mlp_coef = 200000 // control_size
    mlp_coef_val = 500

    seed = int(time()) % 2048

    ser_name = f'treat part experiment.pk'

    nw_cv_grid = {
        'gamma': [10 ** i for i in range(-8, 11)] + [0.5, 5, 50, 100, 200, 500, 700],
    }
    forest_cv_grid = {
        'n_estimators': [10, 50, 100, 300],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_samples_leaf': [1, 0.05, 0.1, 0.2],
    }

    results = {
        'T-NW control mse': [[] for _ in range(len(treat_parts))],
        'T-NW treat mse': [[] for _ in range(len(treat_parts))],
        'T-NW CATE mse': [[] for _ in range(len(treat_parts))],
        'S-NW control mse': [[] for _ in range(len(treat_parts))],
        'S-NW treat mse': [[] for _ in range(len(treat_parts))],
        'S-NW CATE mse': [[] for _ in range(len(treat_parts))],
        'X-NW CATE mse': [[] for _ in range(len(treat_parts))],
        'T-Learner control mse': [[] for _ in range(len(treat_parts))],
        'T-Learner treat mse': [[] for _ in range(len(treat_parts))],
        'T-Learner CATE mse': [[] for _ in range(len(treat_parts))],
        'S-Learner control mse': [[] for _ in range(len(treat_parts))],
        'S-Learner treat mse': [[] for _ in range(len(treat_parts))],
        'S-Learner CATE mse': [[] for _ in range(len(treat_parts))],
        'X-Learner CATE mse': [[] for _ in range(len(treat_parts))],
        'Kernel control': [[] for _ in range(len(treat_parts))],
        'Kernel treat': [[] for _ in range(len(treat_parts))],
        'Kernel CATE': [[] for _ in range(len(treat_parts))],
        'time': [],
    }

    tf.config.set_visible_devices([], 'GPU')
    np.seterr(invalid='ignore')
    np.random.seed(seed)

    try:
        for i in range(iters):
            setup = cur_setup(m)
            for j, treat_part in enumerate(treat_parts):
                seed = np.random.randint(0, 2048)
                print('part = ', treat_part)
                treat_size = int(treat_part * control_size)
                n_t = treat_size // 2
                n_c = n_t
                setup.make_set(control_size, treat_part, test_size)
                start_time = time()

                train_x, train_y, train_w = setup.get_train_set()
                control_x, control_y, treat_x, treat_y = setup.get_cotrol_treat_sets()
                test_x, test_control, test_treat,  test_cate = setup.get_test_set()
                val_set_c, val_labels_c, val_set_t, val_labels_t = setup.get_val_set()

                model = get_basic_dynamic_model(
                    setup, m, n, epochs_num_ordinary, mlp_coef, mlp_coef_val, learning_rate, seed, batch_size, patience)

                *test_data_control, cnt_label = make_spec_set(
                    control_x, test_x, control_y, test_control, control_size, m, 1)
                *test_data_treat, trt_label = make_spec_set(treat_x, test_x, treat_y, test_treat, treat_size, m, 1)
                cnt_pred = model.predict(test_data_control)
                results['Kernel control'][j].append(calc_mse(cnt_pred, cnt_label))

                alpha_model = get_alpha_dynamic_model(
                    setup, m, n_c, n_t, epochs_num_alpha, mlp_coef_val, learning_rate, seed, batch_size, tasks, alpha, patience)
                trt_pred_a = alpha_model.predict(test_data_treat)
                results['Kernel treat'][j].append(calc_mse(trt_pred_a, trt_label))
                results['Kernel CATE'][j].append(
                    calc_mse(trt_pred_a - cnt_pred, trt_label - cnt_label))

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
                        results[f'{key} control mse'][j].append(
                            calc_mse(control_pred, test_control))
                    if (hasattr(instance, 'predict_treat')):
                        treat_pred = instance.predict_treat(test_x)
                        results[f'{key} treat mse'][j].append(calc_mse(treat_pred, test_treat))
                    cate_pred = instance.predict(test_x)
                    results[f'{key} CATE mse'][j].append(calc_mse(cate_pred, test_cate))

                print('itetation ', i + 1, '/', iters)
                cur_time = time() - start_time
                print('time: ', cur_time)
                results['time'].append(cur_time)
                for key, val in results.items():
                    if key != 'time':
                        print(key, ': ', round(val[j][-1], 4))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        pass
    print('time elapsed: ', round(np.sum(results['time']), 0), ' s.')

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
        'tasks': tasks,
        'iters': iters,
        'partititons': treat_parts,
        'treat_size': treat_size,
        'test_size': test_size,
        'seed': seed,
        'learning_rate': learning_rate,
        'val_part_c': 0.2,
        'val_part_t': 0,
        'mlp_coef_val': mlp_coef_val
    }

    results['params'] = params
    with open(f'res_dicts/{ser_name}', 'wb') as file:
        pickle.dump(results, file)
