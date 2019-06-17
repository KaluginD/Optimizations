import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    # TODO: implement.
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        return np.linalg.norm(self.matvec_Ax(x) - self.b) ** 2 / 2

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        return self.matvec_ATx(self.matvec_Ax(x) - self.b)


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    # TODO: implement.
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        """
        Computes the value of h(x).
        """
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + regcoef * ||x||_1 }.
        """
        return np.sign(x) * np.maximum(np.abs(x) - alpha * self.regcoef, 0)


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    # TODO: implement.
    def __init__(self, f, h):
        super().__init__(f, h)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        ATAx_b = self._f.matvec_ATx(Ax_b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self._f.b, self._h.regcoef)


class BarrierLassoOracle(BaseSmoothOracle):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    """
    def __init__(self, A, b, regcoef, t):
        self.A = A
        self.matvec_Ax = lambda x: A @ x
        self.matvec_ATx = lambda x: A.T @ x
        self.b = b
        self.regcoef = regcoef
        self.t = t * 1.0

    def original_func(self, point):
        x, u = np.array_split(point, 2)
        return np.linalg.norm(self.matvec_Ax(x) - self.b) ** 2 / 2 + self.regcoef * np.sum(u)

    def func(self, point):
        x, u = np.array_split(point, 2)
        return self.t * self.original_func(point) - np.sum(np.log(u + x) + np.log(u - x))

    def grad(self, point):
        x, u = np.array_split(point, 2)
        grad_f_x = self.t * self.matvec_ATx(self.matvec_Ax(x) - self.b)
        grad_f_u = self.t * self.regcoef * np.ones(len(u))
        grad_bar_x = -1. / (u + x) + 1. / (u - x)
        grad_bar_u = -1. / (u + x) - 1. / (u - x)
        return np.hstack([grad_f_x + grad_bar_x, grad_f_u + grad_bar_u])

    def hess(self, point):
        x, u = np.array_split(point, 2)
        hess_xx = self.A.T @ self.A * self.t + np.diag(1. / (u - x) ** 2 + 1. / (u + x) ** 2)
        hess_xu = np.diag(1. / (u + x) ** 2 - 1. / (u - x) ** 2)
        hess_uu = np.diag(1. / (u + x) ** 2 + 1. / (u - x) ** 2)
        return np.vstack((np.hstack((hess_xx, hess_xu)), np.hstack((hess_xu, hess_uu))))


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    if np.linalg.norm(ATAx_b, ord=np.inf) < 1e-5:
        mu = Ax_b
    else:
        mu = np.min([1., regcoef / np.linalg.norm(ATAx_b, ord=np.inf)]) * Ax_b
    nu = np.linalg.norm(Ax_b) ** 2 / 2 + regcoef * np.linalg.norm(x, 1) + \
         np.linalg.norm(mu) ** 2 / 2 + b @ mu
    return nu

def create_lasso_prox_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))
