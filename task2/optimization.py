from collections import defaultdict  # Use this for effective implementation of L-BFGS
from datetime import datetime

import numpy as np

from utils import get_line_search_tool


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)

    # TODO: Implement Conjugate Gradients method.
    def fill_history():
        if not trace:
            return
        history['time'].append((datetime.now() - t_0).seconds)
        history['residual_norm'].append(g_k_norm)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def do_display():
        if display:
            print('x = {}, ||Ax - b|| = {}'.format(x_k, g_k_norm))

    t_0 = datetime.now()
    x_size = len(x_k)
    b_norm = np.linalg.norm(b)
    message = None

    g_k = matvec(x_k) - b
    g_k_norm = np.linalg.norm(g_k)
    d_k = -g_k

    max_iter = min(max_iter, 2 * x_size) if max_iter else 2 * x_size
    for _ in range(max_iter):
        do_display()
        fill_history()

        Ad_k = matvec(d_k)
        alpha = (g_k.T @ g_k) / (d_k.T @ Ad_k)
        x_k = x_k + alpha * d_k
        g_k_old = np.copy(g_k)
        g_k = g_k + alpha * Ad_k
        g_k_norm = np.linalg.norm(g_k)
        if g_k_norm <= tolerance * b_norm:
            message = 'success'
            break

        beta = (g_k.T @ g_k) / (g_k_old.T @ g_k_old)
        d_k = -g_k + beta * d_k

    do_display()
    fill_history()

    if not g_k_norm <= tolerance * b_norm:
        message = 'iterations_exceeded'

    return x_k, message, history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    def fill_history():
        if not trace:
            return
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - t_0).seconds)
        history['grad_norm'].append(grad_k_norm)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def do_display():
        if not display:
            return
        if len(x_k) <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, grad_norm = {}'.format(np.round(oracle.func(x_k), 4),
                                                    np.round(grad_k_norm, 4)))

    t_0 = datetime.now()
    x_size = len(x_k)
    message = None

    grad_k = oracle.grad(x_k)
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)

    def bfgs_multiply(v, H, gamma_0):
        if len(H) == 0:
            return gamma_0 * v
        s, y = H[-1]
        H = H[:-1]
        v_new = v - (s @ v) / (y @ s) * y
        z = bfgs_multiply(v_new, H, gamma_0)
        result = z + (s @ v - y @ z) / (y @ s) * s
        return result

    def bfgs_direction():
        if len(H) == 0:
            return -grad_k
        s, y = H[-1]
        gamma_0 = (y @ s) / (y @ y)
        return bfgs_multiply(-grad_k, H, gamma_0)

    H = []
    for k in range(max_iter):
        do_display()
        fill_history()

        d = bfgs_direction()
        alpha = line_search_tool.line_search(oracle, x_k, d)
        x_new = x_k + alpha * d
        grad_new = oracle.grad(x_new)
        H.append((x_new - x_k, grad_new - grad_k))
        if len(H) > memory_size:
            H = H[1:]
        x_k, grad_k = x_new, grad_new
        grad_k_norm = np.linalg.norm(grad_k)
        if grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
            message = 'success'
            break

    do_display()
    fill_history()

    if not grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
        message = 'iterations_exceeded'

    return x_k, message, history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    def fill_history():
        if not trace:
            return
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - t_0).seconds)
        history['grad_norm'].append(grad_k_norm)
        if x_size <= 2:
            history['x'].append(np.copy(x_k))

    def do_display():
        if not display:
            return
        if len(x_k) <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, grad_norm = {}'.format(np.round(oracle.func(x_k), 4),
                                                    np.round(grad_k_norm, 4)))

    t_0 = datetime.now()
    x_size = len(x_k)
    message = None

    grad_k = oracle.grad(x_k)
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)

    for _ in range(max_iter):
        do_display()
        fill_history()

        eps = min(0.5, grad_k_norm ** 0.5)
        while True:
            hess_vec = lambda v: oracle.hess_vec(x_k, v)
            d, _, _ = conjugate_gradients(hess_vec, -grad_k, -grad_k, eps)
            if grad_k @ d < 0:
                break
            else:
                eps *= 10
        alpha = line_search_tool.line_search(oracle, x_k, d, previous_alpha=1)
        x_k = x_k + alpha * d
        grad_k = oracle.grad(x_k)
        grad_k_norm = np.linalg.norm(grad_k)
        if grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
            message = 'success'
            break

    do_display()
    fill_history()
    if not grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
        message = 'iterations_exceeded'

    return x_k, message, history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
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
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.
    def fill_history():
        if not trace:
            return
        history['time'].append((datetime.now() - t_0).seconds)
        history['func'].append(func_k)
        history['grad_norm'].append(grad_k_norm)
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))

    t_0 = datetime.now()
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    a_k = None
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)
    fill_history()
    if display:
        print('Begin new GD')

    for i in range(max_iter):
        if display:
            print('i = {} grad_norm = {} func = {} x = {} grad = {}'.format(i,
                                                                            grad_k_norm,
                                                                            func_k,
                                                                            x_k, grad_k), end=' ')
        if grad_k_norm ** 2 <= tolerance * grad_0_norm ** 2:
            break

        d_k = -grad_k
        a_k = line_search_tool.line_search(oracle, x_k, d_k, 2 * a_k if a_k else None)
        if display:
            print('alpha = {}'.format(a_k))
        x_k += a_k * d_k
        func_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        grad_k_norm = np.linalg.norm(grad_k)
        fill_history()
    if display:
        print()

    if grad_k_norm ** 2 <= tolerance * grad_0_norm ** 2:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history