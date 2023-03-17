import numpy as np

from ex3 import function1, function2
from functions.Function import Function
from optimization.iterations_to_converge_finder import number_of_iterations_to_converge
from optimization.optimize import gradient_descent


def scaling_impl(lr1, function, function_scaled, lr2):
    start = np.array([20, 20])

    iters = number_of_iterations_to_converge(lr1, function, start)
    gradient_descent(lr1, iters, start, function, draw=True)
    print(iters)

    iters = number_of_iterations_to_converge(lr2, function_scaled, start)
    gradient_descent(lr2, iters, start, function_scaled, draw=True)
    print(iters)


if __name__ == '__main__':
    function1_matrix = np.array([[16, 2], [2, 1]])
    print("Cond =", np.linalg.cond(function1_matrix))

    # x' = x  / 4
    function1_scaled = Function(
        lambda x: x[0] ** 2 + x[1] ** 2 + x[0] * x[1],
        lambda x: np.array([
            2 * x[0] + x[1], 2 * x[1] + x[0]
        ])
    )
    scaled_function_matrix = np.array([[1, 0.5], [0.5, 1]])
    print("Cond of scaled function =", np.linalg.cond(scaled_function_matrix))

    lr = lr_new = 0.06
    scaling_impl(lr, function1, function1_scaled, lr_new)

    function2_matrix = np.array([[100, 0], [0, 1]])
    print("Cond =", np.linalg.cond(function2_matrix))

    lr = 0.005
    lr_new = 0.005

    # x' = (x + 2) / 100, y' = y - 1
    function2_scaled = Function(
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: np.array([
            2 * x[0], 2 * x[1]
        ])
    )
    scaled_function_matrix = np.array([[1, 0], [0, 1]])
    print("Cond of scaled function =", np.linalg.cond(scaled_function_matrix))

    scaling_impl(lr, function2, function2_scaled, lr_new)
