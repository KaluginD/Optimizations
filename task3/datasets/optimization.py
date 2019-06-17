from collections import defaultdict
import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import norm, solve
from scipy.linalg import cho_factor, cho_solve
from time import time
from datetime import datetime
import oracles


def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    def fill_history():
        if not trace:
            return
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - t_0).seconds)
        history['duality_gap'].append(duality_gap_k)
        history['iterations'].append(iterations)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def do_display():
        if not display:
            return
        if x_size <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, duality_gap = {}'.format(np.round(oracle.func(x_k), 4),
                                                np.round(duality_gap_k, 4)))

    history = defaultdict(list) if trace else None
    x_k, L_k = np.copy(x_0), L_0
    t_0 = datetime.now()
    x_size = len(x_k)
    iterations = 0
    message = None
    grad_k = oracle.grad(x_k)
    duality_gap_k = oracle.duality_gap(x_k)

    for k in range(max_iter):
        if duality_gap_k < tolerance:
            break

        do_display()
        fill_history()

        while True:
            iterations += 1
            x_new = oracle.prox(x_k - grad_k / L_k, 1 / L_k)
            if oracle._f.func(x_new) > oracle._f.func(x_k) + grad_k @ (x_new - x_k) +\
                                    L_k / 2 * np.linalg.norm(x_new - x_k) ** 2:
                L_k *= 2
            else:
                x_k = x_new
                break

        L_k /= 2
        grad_k = oracle.grad(x_k)
        duality_gap_k = oracle.duality_gap(x_k)

    do_display()
    fill_history()
    message = 'success' if duality_gap_k < tolerance else 'iterations_exceeded'
    return x_k, message, history


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """
    # TODO: Implement
    def fill_history():
        if not trace:
            return
        history['func'].append(func_min)
        history['time'].append((datetime.now() - t_0).seconds)
        history['duality_gap'].append(duality_gap_k)
        history['iterations'].append(iterations)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def do_display():
        if not display:
            return
        if x_size <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, duality_gap = {}'.format(np.round(func_min, 10),
                                                np.round(duality_gap_k, 4)))

    history = defaultdict(list) if trace else None
    x_k, L_k = np.copy(x_0), L_0
    A_k, v_k, y_k = 0, np.copy(x_k), np.copy(x_k)
    sum_diffs_k = 0
    iterations = 0
    t_0 = datetime.now()
    x_size = len(x_k)
    func_min = oracle.func(x_k)
    duality_gap_k = oracle.duality_gap(x_k)
    x_star = np.copy(x_k)

    for k in range(max_iter):
        if duality_gap_k < tolerance:
            break

        do_display()
        fill_history()

        while True:
            iterations += 1

            a_k = (1. + np.sqrt(1. + 4. * L_k * A_k)) / (2. * L_k)
            A_new = A_k + a_k
            y_k = (A_k * x_k + a_k * v_k) / A_new
            grad_k = oracle.grad(y_k)
            sum_diffs_new = sum_diffs_k + a_k * grad_k
            v_new = oracle.prox(x_0 - sum_diffs_new, A_new)
            x_new = (A_k * x_k + a_k * v_new) / A_new

            func_x = oracle.func(x_new)
            func_y = oracle.func(y_k)
            fs = np.array([func_min, func_x, func_y])
            func_min = np.min(fs)
            if func_min == func_x:
                x_star = x_new
            if func_min == func_y:
                x_star = y_k

            if oracle._f.func(x_new) > oracle._f.func(y_k) + grad_k @ (x_new - y_k) + \
                                    L_k / 2 * np.linalg.norm(x_new - y_k) ** 2:
                L_k *= 2
            else:
                x_k = x_new
                A_k = A_new
                v_k = v_new
                sum_diffs_k = sum_diffs_new
                break

        L_k /= 2
        duality_gap_k = oracle.duality_gap(x_star)

    do_display()
    fill_history()
    message = 'success' if duality_gap_k < tolerance else 'iterations_exceeded'
    return x_star, message, history


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If callable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    def fill_history():
        if not trace:
            return
        history['func'].append(oracle.original_func(np.concatenate([x_k, u_k])))
        history['time'].append((datetime.now() - time_0).seconds)
        history['duality_gap'].append(duality_gap_k)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def do_display():
        if not display:
            return
        if x_size <= 3:
            print('x = {}, u = {}'.format(np.round(x_k, 4), np.round(u_k, 4)), end='')
        print('func= {}, duality_gap = {}'.format(np.round(oracle.original_func(np.concatenate([x_k, u_k])), 10),
                                                np.round(duality_gap_k, 10)))

    oracle = oracles.BarrierLassoOracle(A, b, reg_coef, t_0)
    lasso_duality_gap = lasso_duality_gap if lasso_duality_gap else oracles.lasso_duality_gap
    lasso_duality_gap_ = lambda x_: lasso_duality_gap(x_, A @ x_ - b, A.T @ (A @ x_ - b), b, reg_coef)

    history = defaultdict(list) if trace else None
    x_k, u_k = np.copy(x_0), np.copy(u_0)
    time_0 = datetime.now()
    x_size = len(x_k)
    t_k = t_0
    duality_gap_k = lasso_duality_gap_(x_k)
    message = None

    for k in range(max_iter):
        if duality_gap_k < tolerance:
            break

        do_display()
        fill_history()
        oracle.t = t_k
        x = np.concatenate([x_k, u_k])
        x_new, message_newton, _ = newton(oracle, x, max_iter=max_iter_inner, tolerance=tolerance_inner,
                                        line_search_options={'c1' : c1})
        x_k, u_k = np.array_split(x_new, 2)
        if message_newton == 'computational_error':
            message = message_newton
            break

        t_k *= gamma
        duality_gap_k = lasso_duality_gap_(x_k)

    do_display()
    fill_history()
    if not message:
        message = 'success' if duality_gap_k < tolerance else 'iterations_exceeded'
    return (x_k, u_k), message, history

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.
        Method of tuning step-size.
    method == 'Armijo':
        c1 : Constant for Armijo rule
        alpha_0 : Starting point for the backtracking procedure.
    """
    def __init__(self, c1=1e-4, alpha_0=1.):
        self.c1 = c1
        self.alpha_0 =alpha_0

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        phi = lambda a: oracle.func_directional(x_k, d_k, a)
        derphi = lambda a: oracle.grad_directional(x_k, d_k, a)
        phi0, derphi0 = phi(0), derphi(0)

        alpha = previous_alpha if previous_alpha else self.alpha_0
        while phi(alpha) > phi0 + self.c1 * alpha * derphi0:
            alpha /= 2
        return alpha

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    def fill_history():
        if not trace:
            return
        history['time'].append(datetime.now() - t_0)
        history['func'].append(func_k)
        history['grad_norm'].append(grad_k_norm)
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))

    def get_alpha(x_concat, d_concat):
        x, u = np.array_split(x_concat, 2)
        grad_x, grad_u = np.array_split(d_concat, 2)
        alphas = [1.]
        THETA = 0.99
        for i in range(len(grad_x)):
            if grad_x[i] > grad_u[i]:
                alphas.append(THETA * (u[i] - x[i]) / (grad_x[i] - grad_u[i]))
            if grad_x[i] < -grad_u[i]:
                alphas.append(THETA * (x[i] + u[i]) / (-grad_x[i] - grad_u[i]))
        return min(alphas)

    t_0 = datetime.now()
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    hess_k = oracle.hess(x_k)
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)
    fill_history()
    if display:
        print('Begin new NM')

    for i in range(max_iter):
        if display:
            print('i = {} grad_norm = {} func = {} x = {} grad = {}'.format(i, grad_k_norm, func_k, x_k, grad_k))
        if grad_k_norm ** 2 <= tolerance * grad_0_norm ** 2:
            break
        try:
            d_k = cho_solve(cho_factor(hess_k), -grad_k)
        except LinAlgError:
            return x_k, 'computational_error', history

        a_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=get_alpha(x_k, d_k))
        x_k += a_k * d_k
        func_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        hess_k = oracle.hess(x_k)
        grad_k_norm = np.linalg.norm(grad_k)
        fill_history()

    if grad_k_norm ** 2 <= tolerance * grad_0_norm ** 2:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history