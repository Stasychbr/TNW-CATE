import pickle
import matplotlib.pyplot as plt
import numpy as np

format = {
    'T-NW': {'marker': 'v', 'linestyle': (0, (1, 4)), 'markerfacecolor': 'white'},
    'S-NW': {'marker': '^', 'linestyle': (0, (1, 5)), 'markerfacecolor': 'white'},
    'X-NW': {'marker': 'o', 'linestyle': (0, (1, 5)), 'markerfacecolor': 'white'},
    'T-RF': {'marker': 'v', 'linestyle': (0, (5, 5)), 'markerfacecolor': 'white'},
    'S-RF': {'marker': '^', 'linestyle': (0, (5, 5)), 'markerfacecolor': 'white'},
    'X-RF': {'marker': 'o', 'linestyle': (0, (5, 5)), 'markerfacecolor': 'white'},
    'Kernel': {'marker': 'X', 'markerfacecolor': 'white'},
}


def draw_alpha_test(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    alpha = [0] + d['params']['alpha']
    k_treat = [np.mean(d['Alpha_0 treat'])] + [np.mean(l) for l in d['Alpha treat']]
    k_cate = [np.mean(d['Alpha_0 CATE'])] + [np.mean(l) for l in d['Alpha CATE']]
    treat = {
        'T-NW': np.mean(d['T-NW treat mse']),
        'S-NW': np.mean(d['S-NW treat mse']),
        'T-RF': np.mean(d['T-Learner treat mse']),
        'X-RF': np.mean(d['S-Learner treat mse']),
    }
    cate = {
        'T-NW': np.mean(d['T-NW CATE mse']),
        'S-NW': np.mean(d['S-NW CATE mse']),
        'X-NW': np.mean(d['X-NW CATE mse']),
        'T-RF': np.mean(d['T-Learner CATE mse']),
        'S-RF': np.mean(d['S-Learner CATE mse']),
        'X-RF': np.mean(d['X-Learner CATE mse']),
    }
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('MSE')
    ax.set_xticks(alpha)
    ax.set_title('Treatment')
    ax.semilogy(alpha, k_treat, **format['Kernel'], label='TNW-CATE')
    for key, val in treat.items():
        ax.semilogy(alpha, [val] * len(alpha), **format[key], label=key)
    ax.legend()
    ax.grid()
    fig.savefig(f'Alpha_treatment_{func}.pdf')
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 4.8), dpi=300)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('MSE')
    ax.set_xticks(alpha)
    ax.set_title('CATE')
    ax.semilogy(alpha, k_cate, **format['Kernel'], label='TNW-CATE')
    for key, val in cate.items():
        ax.semilogy(alpha, [val] * len(alpha), **format[key], label=key)
    ax.legend()
    ax.grid()
    fig.savefig(f'Alpha_CATE_{func}.pdf')


def draw_size_test(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    sizes = d['params']['control_sizes']
    k_treat = [np.mean(l) for l in d['Kernel treat']]
    k_cate = [np.mean(l) for l in d['Kernel CATE']]
    k_control = [np.mean(l) for l in d['Kernel control']]
    treat = {
        'T-NW': [np.mean(l) for l in d['T-NW treat mse']],
        'S-NW': [np.mean(l) for l in d['S-NW treat mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner treat mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner treat mse']],
    }
    control = {
        'T-NW': [np.mean(l) for l in d['T-NW control mse']],
        'S-NW': [np.mean(l) for l in d['S-NW control mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner control mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner control mse']],
    }
    cate = {
        'T-NW': [np.mean(l) for l in d['T-NW CATE mse']],
        'S-NW': [np.mean(l) for l in d['S-NW CATE mse']],
        'X-NW': [np.mean(l) for l in d['X-NW CATE mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner CATE mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner CATE mse']],
        'X-RF': [np.mean(l) for l in d['X-Learner CATE mse']],
    }
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('control size')
    ax.set_ylabel('MSE')
    ax.set_xticks(sizes)
    ax.set_title('Treatment')
    ax.semilogy(sizes, k_treat, **format['Kernel'], label='TNW-CATE')
    for key, val in treat.items():
        ax.semilogy(sizes, val, **format[key], label=key)
    ax.legend()
    ax.grid()
    fig.savefig(f'Size_treatment_{func}.pdf')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel('control size')
    ax.set_ylabel('MSE')
    ax.set_xticks(sizes)
    ax.set_title('CATE')
    ax.semilogy(sizes, k_cate, **format['Kernel'], label='TNW-CATE')
    for key, val in cate.items():
        ax.semilogy(sizes, val, **format[key], label=key)
    ax.legend()
    ax.grid()
    fig.savefig(f'Size_CATE_{func}.pdf')
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('control size')
    ax.set_ylabel('MSE')
    ax.set_xticks(sizes)
    ax.set_title('Control')
    ax.semilogy(sizes, k_control, **format['Kernel'], label='TNW-CATE')
    for key, val in control.items():
        ax.semilogy(sizes, val, **format[key], label=key)
    ax.legend()
    ax.grid()
    fig.savefig(f'Size_control_{func}.pdf')


def draw_part_test(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    partitions = d['params']['partititons']
    part_lab = [f'{int(100 * x)}%' for x in partitions]
    k_treat = [np.mean(l) for l in d['Kernel treat']]
    k_cate = [np.mean(l) for l in d['Kernel CATE']]
    k_control = [np.mean(l) for l in d['Kernel control']]
    treat = {
        'T-NW': [np.mean(l) for l in d['T-NW treat mse']],
        'S-NW': [np.mean(l) for l in d['S-NW treat mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner treat mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner treat mse']],
    }
    control = {
        'T-NW': [np.mean(l) for l in d['T-NW control mse']],
        'S-NW': [np.mean(l) for l in d['S-NW control mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner control mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner control mse']],
    }
    cate = {
        'T-NW': [np.mean(l) for l in d['T-NW CATE mse']],
        'S-NW': [np.mean(l) for l in d['S-NW CATE mse']],
        'X-NW': [np.mean(l) for l in d['X-NW CATE mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner CATE mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner CATE mse']],
        'X-RF': [np.mean(l) for l in d['X-Learner CATE mse']],
    }
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('treatment part')
    ax.set_ylabel('MSE')
    ax.set_xticks(partitions)
    ax.set_xticklabels(part_lab)
    ax.set_title('Treatment')
    ax.semilogy(partitions, k_treat, **format['Kernel'], label='TNW-CATE')
    for key, val in treat.items():
        ax.semilogy(partitions, val, label=key, **format[key])
    ax.legend()
    ax.grid()
    fig.savefig(f'Part_treatment_{func}.pdf')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    ax.set_xlabel('treatment part')
    ax.set_ylabel('MSE')
    ax.set_xticks(partitions)
    ax.set_xticklabels(part_lab)
    ax.set_title('CATE')
    ax.semilogy(partitions, k_cate, **format['Kernel'], label='TNW-CATE')
    for key, val in cate.items():
        ax.semilogy(partitions, val, label=key, **format[key])
    ax.legend()
    ax.grid()
    fig.savefig(f'Part_CATE_{func}.pdf')
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('treatment part')
    ax.set_ylabel('MSE')
    ax.set_xticks(partitions)
    ax.set_xticklabels(part_lab)
    ax.set_title('Control')
    ax.semilogy(partitions, k_control, **format['Kernel'], label='TNW-CATE')
    for key, val in control.items():
        ax.semilogy(partitions, val, **format[key], label=key)
    ax.legend()
    ax.grid()
    fig.savefig(f'Part_control_{func}.pdf')


def draw_real_part_test(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    partitions = d['params']['train_parts']
    part_lab = [f'{int(100 * x)}%' for x in partitions]
    k_treat = [np.mean(l) for l in d['Kernel treat']]
    k_cate = [np.mean(l) for l in d['Kernel CATE']]
    k_control = [np.mean(l) for l in d['Kernel control']]
    treat = {
        'T-NW': [np.mean(l) for l in d['T-NW treat mse']],
        'S-NW': [np.mean(l) for l in d['S-NW treat mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner treat mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner treat mse']],
    }
    control = {
        'T-NW': [np.mean(l) for l in d['T-NW control mse']],
        'S-NW': [np.mean(l) for l in d['S-NW control mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner control mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner control mse']],
    }
    cate = {
        'T-NW': [np.mean(l) for l in d['T-NW CATE mse']],
        'S-NW': [np.mean(l) for l in d['S-NW CATE mse']],
        'X-NW': [np.mean(l) for l in d['X-NW CATE mse']],
        'T-RF': [np.mean(l) for l in d['T-Learner CATE mse']],
        'S-RF': [np.mean(l) for l in d['S-Learner CATE mse']],
        'X-RF': [np.mean(l) for l in d['X-Learner CATE mse']],
    }
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('train part')
    ax.set_ylabel('MSE')
    ax.set_xticks(partitions)
    ax.set_xticklabels(part_lab)
    ax.set_title('Treatment MSE for IHDP dataset')
    ax.semilogy(partitions, k_treat, **format['Kernel'], label='TNW-CATE')
    for key, val in treat.items():
        ax.semilogy(partitions, val, label=key, **format[key])
    ax.legend()
    ax.grid()
    fig.savefig(f'real_treatment_{func}.pdf')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    ax.set_xlabel('train part')
    ax.set_ylabel('MSE')
    ax.set_xticks(partitions)
    ax.set_xticklabels(part_lab)
    ax.set_title('CATE MSE for IHDP dataset')
    ax.semilogy(partitions, k_cate, **format['Kernel'], label='TNW-CATE')
    for key, val in cate.items():
        ax.semilogy(partitions, val, label=key, **format[key])
    ax.legend()
    ax.grid()
    fig.savefig(f'real_CATE_{func}.pdf')
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('train part')
    ax.set_ylabel('MSE')
    ax.set_xticks(partitions)
    ax.set_xticklabels(part_lab)
    ax.set_title('Control MSE for IHDP dataset')
    ax.semilogy(partitions, k_control, **format['Kernel'], label='TNW-CATE')
    for key, val in control.items():
        ax.semilogy(partitions, val, **format[key], label=key)
    ax.legend()
    ax.grid()
    fig.savefig(f'real_control_{func}.pdf')


func = 'suffix_name'  # only for filename
# draw_alpha_test('res_dicts/alpha experiment.pk')
draw_part_test('res_dicts/treat part experiment.pk')
# draw_size_test('res_dicts/size experiment.pk')
# draw_real_part_test('res_dicts/ihdp experiment.pk')
