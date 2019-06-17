import oracles
import optimization

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file
from datetime import datetime
from scipy.sparse import diags

np.random.seed(seed=42)

def first_experiment():
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
                matvec = lambda x: A @ x
                x_star, msg, history = optimization.conjugate_gradients(matvec, b, np.zeros(n),
                                                                     trace=True)
                if msg == 'success':
                    T[n][i].append(len(history['time']))
                else:
                    T[n][i].append(10000)
            plt.plot(kappas, T[n][i], ls='--', color=color, alpha=0.2)
        plt.plot(kappas, np.mean(T[n], axis=0), color=color, label='n = {}'.format(n))
        print('n = {}, time = {}'.format(n, datetime.now() - t0))

    plt.grid()
    plt.legend()
    plt.ylabel('iterations')
    plt.xlabel(r'$\ae$')
    plt.savefig('experiment_1/T(n, kappa)')


def second_experiment():
    data_path = 'data/gisette_scale'
    result_path = lambda x: 'experiment_2/grad_norm-vs-{}'.format(x)
    A, b = load_svmlight_file(data_path)
    m, n = A.shape
    oracle = oracles.create_log_reg_oracle(A, b, 1 / m)
    results = []
    ls = [0, 1, 5, 10, 50, 100]
    for l in ls:
        _, _, history = optimization.lbfgs(oracle, np.zeros(n), memory_size=l, trace=True)
        print('lbfgs with l = {} finished'.format(l))
        grad_norm = np.array(history['grad_norm'])
        grad_norm /= grad_norm[0]
        grad_norm = np.power(grad_norm, 2)
        grad_norm = np.log(grad_norm)
        results.append((l, grad_norm, history['time']))

    def plotting(flag):
        plt.figure(figsize=(12, 8))
        for l, grad_norm, times in results:
            x = list(range(len(grad_norm))) if flag == 'iterations' else times
            plt.plot(x, grad_norm, label='history size = {}'.format(l))
        plt.xlabel('iterations' if flag == 'iterations' else 'seconds')
        plt.ylabel(r'$\log\left(grad\_norm\right)$')
        plt.legend()
        plt.grid()
        plt.savefig(result_path(flag))

    plotting('seconds')
    plotting('iterations')


def third_experiment():
    data_path = lambda name: 'data/{}'.format(name)
    datasets = [
        'gisette_scale',
        'news20.binary',
        'rcv1_train.binary',
        'real-sim',
        'w8a'
    ]

    algorithms = [
        optimization.hessian_free_newton,
        optimization.lbfgs,
        optimization.gradient_descent
    ]

    def plotting(hfn_history, lbfgs_history, gd_history, dataset):
        figname = lambda data, x, y: 'experiment_3/{}_{}-vs-{}.png'.format(data, y, x)

        def get_x(history, form):
            if form == 'iterations':
                return list(range(len(history['time'])))
            if form == 'time':
                return history['time']

        def get_y(history, form):
            if form == 'func':
                return history['func']
            if form == 'grad':
                grad_norm = np.array(history['grad_norm'])
                grad_norm /= grad_norm[0]
                grad_norm = np.power(grad_norm, 2)
                grad_norm = np.log(grad_norm)
                return grad_norm

        histories = [hfn_history, lbfgs_history, gd_history]
        names = ['HFN', 'L-BFGS', 'GD']
        colors = ['b', 'r', 'g']
        for x_form in ['iterations', 'time']:
            for y_form in ['func', 'grad']:
                if (x_form, y_form) == ('iterations', 'grad'):
                    continue
                plt.figure()
                for history, name, color in zip(histories, names, colors):
                    plt.plot(get_x(history, x_form), get_y(history, y_form), label=name, color=color)
                plt.title(dataset)
                plt.xlabel(x_form)
                plt.ylabel(y_form if y_form == 'func' else r'$\log\left(grad\_norm\right)$')
                plt.grid()
                plt.legend()
                to_save = figname(dataset, x_form, y_form)
                plt.savefig(to_save)

    for dataset in datasets:
        print(dataset)
        A, b = load_svmlight_file(data_path(dataset))
        m, n = A.shape
        oracle = oracles.create_log_reg_oracle(A, b, 1 / m)
        histories = []
        for i, algorithm in enumerate(algorithms):
            _, _, history = algorithm(oracle, np.zeros(n), trace=True)
            print('{} algo finished'.format(i))
            histories.append(history)
        plotting(*histories, dataset)


if __name__ == '__main__':
    first_experiment()
    second_experiment()
    third_experiment()
