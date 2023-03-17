import numpy as np
import matplotlib.pyplot as plt

from functions.Function import Function
from optimization.optimize import gradient_descent

plt.rcParams["figure.figsize"] = (10, 10)

function1 = Function(
    lambda x: 16 * x[0] ** 2 + x[1] ** 2 + 4 * x[0] * x[1],
    lambda x: np.array([
        32 * x[0] + 4 * x[1], 2 * x[1] + 4 * x[0]
    ])
)

function2 = Function(
    lambda x: 100 * (x[0] - 2) ** 2 + (x[1] + 1) ** 2,
    lambda x: np.array([
        200 * (x[0] - 2), 2 * (x[1] + 1)
    ])
)


def ex3():
    start = np.array([20, 20])

    def ex3_impl(lr, function, epoch, start_point):
        gradient_descent(lr, epoch, start_point, function, logging=True)
        gradient_descent(lr, epoch, start_point, function, logging=True, linear_search="dichotomy")

    lr = 0.05
    epoch = 20
    ex3_impl(lr, function1, epoch, start)

    lr = 0.001
    epoch = 30
    ex3_impl(lr, function2, epoch, start)


if __name__ == '__main__':
    ex3()
