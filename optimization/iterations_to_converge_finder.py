import numpy as np

from optimization.optimize import grad_descent_once


def number_of_iterations_to_converge(lr, f, start, linear_search=None, eps=0.1):
    count = 0

    while (np.linalg.norm(f.grad(start)) > eps) and count < 10000:
        start = grad_descent_once(f, lr, start, linear_search=linear_search)
        count += 1

    return count
