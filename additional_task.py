import random

import numpy as np

from functions.Function import Function
from optimization.iterations_to_converge_finder import number_of_iterations_to_converge
from optimization.optimize import gradient_descent
from ex3 import function1, function2


def compare_methods_impl(lr, function, start, linear_search=None):
    num = number_of_iterations_to_converge(lr, function, start, linear_search=linear_search) + 1
    print("Iters to converge:", num)

    function.reset_calc_amount()
    gradient_descent(lr, num, start, function, draw=False, linear_search=linear_search)
    print("Function calc amount:", function.function_calc_amount)
    print("Grad calc amount:", function.grad_calc_amount)


def compare_methods(lr, function):
    start = np.array([20, 20])

    print("Constant step")
    compare_methods_impl(lr, function, start)
    print()

    print("Dichotomy")
    compare_methods_impl(lr, function, start, linear_search="dichotomy")
    print()

    print("Wolfe")
    compare_methods_impl(lr, function, start, linear_search="wolfe")
    print()


if __name__ == '__main__':
    compare_methods(0.06, function1)

    compare_methods(0.005, function2)

    low_condition_function = Function(
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: np.array([2 * x[0], 2 * x[1]])
    )

    compare_methods(0.15, low_condition_function)
