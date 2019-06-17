import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file
import itertools

import optimization
import oracles


def first_experiment():
    A, b = load_svmlight_file('datasets/w8a')
    regcoef = 1
    oracle = oracles.create_lasso_prox_oracle(A, b, regcoef)
    m, n = A.shape
    x_0 = np.zeros(n)

    def plot_iterations(history, name):
        name_ = name
        name = ' '.join(name.split('_')).title()
        y = history['iterations']
        x = np.array(list(range(len(y))))
        plt.figure()
        plt.title(name + '\nmin duality gap = {}'.format(round(min(history['duality_gap']), 2)), fontsize=14)
        plt.plot(x, y, label='line search iterations')
        plt.plot(x, 2 * x, label='double iterations')
        plt.xlabel('method iterations', fontsize=14)
        plt.ylabel('line search iterations', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
        plt.savefig('experiment_1/' + name_ + '.png')

    for method in [optimization.proximal_gradient_method,
                   optimization.proximal_fast_gradient_method]:
        print(method.__name__)
        x_star, message, history = method(oracle, x_0, trace=True)
        plot_iterations(history, method.__name__)


def second_experiment():
    ns = [50, 100, 500]
    ms = [1000, 5000, 10000]
    regcoefs = [0.1, 1, 10]
    methods = [optimization.proximal_gradient_method,
               optimization.proximal_fast_gradient_method,
               optimization.barrier_method_lasso]

    def plot_accuracy(histories, n, m, regcoef):
        name = 'n={}, m={}, regcoef={}'.format(n, m, regcoef)
        for x_name in ['iterations', 'time']:
            plt.figure()
            plt.title(name)
            for history, method in zip(histories, methods):
                method_name = ' '.join(method.__name__.split('_')).title()
                y = history['duality_gap']
                x = np.array(history['time']) * 1000 if x_name == 'time' else list(range(len(y)))
                plt.plot(x, y, label=method_name)
            plt.xlabel(x_name, fontsize=14)
            plt.ylabel('duality gap', fontsize=14)
            plt.yscale('log')
            plt.legend(fontsize=10, loc=1)
            plt.grid()
            plt.savefig('experiment_2/' + name + ' ' + x_name + '.png')
        plt.close('all')

    for iteration, (n, m, regcoef) in enumerate(itertools.product(ns, ms, regcoefs)):
        print(n, m, regcoef)
        np.random.seed(iteration * 50 + 42)
        A = np.random.uniform(low=-100, high=100, size=(m, n))
        b = np.random.uniform(low=-100, high=100, size=m)
        x_0 = np.zeros(n)
        oracle = oracles.create_lasso_prox_oracle(A, b, regcoef)
        histories = []
        for i, method in enumerate(methods):
            print(method.__name__)
            if i < 2:
                x_star, message, history = method(oracle, x_0, trace=True)
            else:
                u_0 = np.array([100] * n)
                x_star, message, history = method(A, b, regcoef, x_0, u_0, trace=True)
            histories.append(history)
        plot_accuracy(histories, n, m, regcoef)


if __name__ == '__main__':
    first_experiment()
    second_experiment()
