import oracles
import numpy as np


def main():
    danger = []
    for _ in range(10):
        A = np.random.uniform(0, 1000, (5, 5))
        b = np.random.uniform(0, 1000, 5)
        regcoef = np.random.uniform(0, 100, 1)
        oracle = oracles.create_log_reg_oracle(A, b, regcoef)
        diffs = []
        for i in range(100):
            x = np.random.uniform(0, 100, 5)
            v = np.random.uniform(0, 100, 5)
            hess_vec_finite = oracles.hess_vec_finite_diff(oracle.func, x, v)
            hess_vec_oracle = oracle.hess_vec(x, v)
            diff = np.abs(hess_vec_finite - hess_vec_oracle)
            if max(diff) > 1:
                danger.append((A, b, regcoef, x, v))
            diffs.append(max(diff))
        print(max(diffs))
    print(len(danger))

if __name__ == '__main__':
    main()