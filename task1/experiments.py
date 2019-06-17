from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from sklearn.datasets import load_svmlight_file

import optimization
import oracles
import plot_trajectory_2d


def first_experiment():
    A_good = np.array([
        [1, 0.2],
        [0.2, 1.2]
    ])
    A_bad = np.array([
        [0.1, 0.1],
        [0.1, 1]
    ])

    for i, A in enumerate([A_good, A_bad]):
        cond = np.linalg.cond(A)
        oracle = oracles.QuadraticOracle(A, np.zeros(2))
        np.random.seed(i * 42)
        x_0_s = [np.random.uniform(-5, 5, size=2) for _ in range(3)]
        for j in range(3):
            x_0 = x_0_s[j]
            for method in ['Wolfe', 'Armijo', 'Constant']:
                _, _, history = optimization.gradient_descent(oracle, x_0,
                                                                       line_search_options={'method': method,
                                                                                            'alpha_0': 100},
                                                                       trace=True)
                plot_trajectory_2d.plt.figure()
                plot_trajectory_2d.plot_levels(oracle.func)
                name = 'experiment_1/{}-{}-{}'.format(round(cond, 3), method, j)
                plot_trajectory_2d.plot_trajectory(oracle.func, history['x'], save=name)
                print('cond = {}, j = {}, method = {}, steps = {}'.format(round(cond, 3), j, method, len(history['x'])))


def second_experiment():
    ns = [10, 100, 1000, 10000]
    colors = ['g', 'r', 'b', 'y']
    kappas = list(range(1, 1000, 100))
    iterations = 10
    T = {}
    np.seterr(all='print')
    t0 = datetime.now()
    for n, color in zip(ns, colors):
        T[n] = [[] for _ in range(iterations)]
        for i in range(iterations):
            for kappa in kappas:
                np.random.seed(1000 * i + kappa)
                diag = np.random.uniform(low=1, high=kappa, size=n)
                diag[0], diag[-1] = 1, kappa
                A = diags(diag)
                b = np.random.uniform(low=1, high=kappa, size=n)
                oracle = oracles.QuadraticOracle(A, b)
                x_star, msg, history = optimization.gradient_descent(oracle, np.zeros(n),
                                                                     trace=True)
                if msg == 'success':
                    T[n][i].append(len(history['grad_norm']))
                else:
                    T[n][i].append(10000)
            plt.plot(kappas, T[n][i], ls='--', color=color, alpha=0.2)
        plt.plot(kappas, np.mean(T[n], axis=0), color=color, label='n = {}'.format(n))
        print('n = {}, time = {}'.format(n, datetime.now() - t0))

    plt.grid()
    plt.legend()
    plt.ylabel('iterations')
    plt.xlabel(r'$\ae$')
    plt.savefig('experiment_2/T(n, kappa)')


def third_experiment():
    data_path = 'experiment_3/datasets/'
    result_path = 'experiment_3/'
    names = [
        'w8a',
        'gisette_scale',
        'real-sim'
    ]
    def plotting(history_gd, history_nm, param):
        f_gd = np.array(history_gd[param])
        f_nm = np.array(history_nm[param])
        time_gd = list(map(lambda i: i.total_seconds(), history_gd['time']))
        time_nm = list(map(lambda i: i.total_seconds(), history_nm['time']))
        if param == 'grad_norm':
            f_gd = np.log(f_gd / f_gd[0])
            f_nm = np.log(f_nm / f_nm[0])
        plt.figure()
        plt.plot(time_gd, f_gd, label='GD')
        plt.plot(time_nm, f_nm, label='Newton')
        plt.xlabel('seconds')
        ylabel = 'func' if param == 'func' else r'$\log\left(grad\_norm\right)$'
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.savefig(result_path + name + '-' + param)


    for name in names:
        A, b = load_svmlight_file(data_path + name)
        m, n = A.shape
        oracle = oracles.create_log_reg_oracle(A, b, 1 /m)
        if name != 'real-sim':
            print('begin')
            x_star_nm, _, history_nm = optimization.newton(oracle, np.zeros(n), trace=True)
            print('Newton is finished')
        x_star_gd, _, history_gd = optimization.gradient_descent(oracle, np.zeros(n), trace=True)
        print('GD is finished')
        plotting(history_gd, history_nm, 'func')
        plotting(history_gd, history_nm, 'grad_norm')


if __name__ == '__main__':
    first_experiment()
    second_experiment()
    third_experiment()
