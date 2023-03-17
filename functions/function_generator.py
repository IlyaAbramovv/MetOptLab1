import numpy as np
from functions.Function import Function


def generate_function(n, k):
    matrix = np.random.randint(-100, 100, (n, n))
    S = np.diag(np.linspace(k, 1, n))
    Q, R = np.linalg.qr(matrix)
    matrix = np.dot(np.dot(Q, S), np.transpose(Q))

    f = lambda x: np.dot(np.dot(np.transpose(x), matrix), x)
    h = 1e-5
    grad = lambda x: (f(x + h * np.eye(n)) - f(x - h * np.eye(n))) / (2 * h)

    return Function(f, grad)
