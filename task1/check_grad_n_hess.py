import oracles
import numpy as np


def main():
    A = np.random.uniform(0, 10, (5, 5))
    b = np.random.uniform(0, 10, 5)
    regcoef = np.random.uniform(0, 10, 1)
    oracle = oracles.create_log_reg_oracle(A, b, regcoef)
    print(A)
    print(b)
    print(regcoef)
    for i in range(10):
        x = np.random.uniform(0, 10, 5)
        grad_oracle = oracle.grad(x)
        hess_oracle = oracle.hess(x)
        grad_finite = oracles.grad_finite_diff(oracle.func, x)
        hess_finite = oracles.hess_finite_diff(oracle.func, x)
        diff_grad = np.abs(grad_finite - grad_oracle)
        diff_hess = np.abs(hess_finite - hess_oracle)
        #print(i)
        #print(grad_oracle)
        #print(grad_finite)
        print(np.max(diff_hess))


if __name__ == '__main__':
    main()