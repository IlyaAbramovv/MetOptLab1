import random

import numpy as np

from functions.function_generator import generate_function
from optimization.iterations_to_converge_finder import number_of_iterations_to_converge
from functools import reduce

if __name__ == '__main__':
    TEST_NUMBER = 5
    arr = [[] for _ in range(TEST_NUMBER)]

    n_range = [2 ** i for i in range(1, 6)]
    k_range = [2 ** i for i in range(1, 11)]

    for i in range(TEST_NUMBER):

        for n in n_range:
            print(n)
            start_point = np.array([random.randint(-100, 100) for _ in range(n)])
            for k in k_range:
                func = generate_function(n, k)

                lr = 1 / (k * 1.5)
                iterations_to_converge: int = number_of_iterations_to_converge(
                    lr, func, start_point, linear_search=None)
                arr[i].append(iterations_to_converge)
                print(iterations_to_converge)


    def sum_elements(x, y):
        ans = []
        for i in range(len(x)):
            ans.append(x[i] + y[i])
        return ans

    arr = map(lambda x: x / TEST_NUMBER, reduce(sum_elements, arr))

    FILE_NAME = "data/ex6data_wolfe.csv"
    with open(FILE_NAME, 'w') as file:
        for n in n_range:
            for k in k_range:
                file.write(f"{n}, {k}, {next(arr)}\n")
