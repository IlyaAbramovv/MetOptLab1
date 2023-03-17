import numpy as np
from matplotlib import pyplot as plt

from functions.Function import Function


def gradient_descent(lr, epoch, start, function, linear_search=None, logging=False, draw=False, save_image=False):
    points = np.zeros((epoch, len(start)))
    points[0] = start
    for i in range(1, epoch):
        start = grad_descent_once(function, lr, start, linear_search=linear_search)
        points[i] = start

    if logging:
        print(points)
    if draw:
        t = np.linspace(-50, 50, 1000)
        X, Y = np.meshgrid(t, t)
        plt.plot(points[:, 0], points[:, 1], 'o-')
        plt.contour(X, Y, function.calc(np.array([X, Y])), levels=sorted([function.calc(p) for p in points]))
        if save_image.__class__ is str:
            plt.savefig(save_image)
        plt.show()

    return start


def grad_descent_once(function, lr, start, linear_search=None):
    grad = function.grad(start)
    right = start - lr * np.array(grad)

    match linear_search:
        case "dichotomy":
            epoch = 5
            return dichotomy(start, right, grad, epoch, lr / (2 ** epoch), function)
        case "wolfe":
            return wolfe(start, function)
        case _:
            return right


def dichotomy(left, right, grad, epoch, lr, function):
    for _ in range(epoch):
        mid = (left + right) / 2
        mid1 = mid + grad * lr / 2  # closer to l
        mid2 = mid - grad * lr / 2  # closer to r
        value1 = function.calc(mid1)
        value2 = function.calc(mid2)
        if value1 > value2:
            left = mid
        else:
            right = mid
    return right


def wolfe(start, func: Function, c1=10e-4, c2=0.9, tau=0.5):
    f = func.calc(start)
    grad = func.grad(start)
    p = -grad
    phi = np.dot(grad, p)
    alpha = 1

    while True:
        x = start + alpha * p
        if func.calc(x) > f + c1 * alpha * phi or np.dot(func.grad(x), p) < c2 * phi:
            alpha *= tau
        else:
            return x
