import random

import numpy as np

from optimization.iterations_to_converge_finder import number_of_iterations_to_converge
from optimization.optimize import gradient_descent
from ex3 import function1, function2


def ex4a():
    start_point = np.array([20, 20])

    def ex4a_impl(learning_rates, function, eps=0.1):
        for lr in learning_rates:
            iter_num = number_of_iterations_to_converge(lr, function, start_point, eps=eps)
            gradient_descent(lr, iter_num, start_point, function, draw=True)
            print(iter_num)

    ex4a_impl([0.007, 0.06, 0.1], function1)
    ex4a_impl([0.001, 0.005, 0.1], function2, 1)


def ex4b():
    start_point = np.array([20, 20])

    def ex4b_impl(lr, function, function_number, linear_search=None):
        epoch = number_of_iterations_to_converge(lr, function, start_point, linear_search=linear_search)
        function.reset_calc_amount()
        gradient_descent(lr, epoch, start_point, function, linear_search=linear_search, draw=True,
                         save_image=f"ex4data/ex4b/f{function_number}{linear_search}.png")
        print(function.function_calc_amount)
        print(function.grad_calc_amount)

    lr = 0.06
    ex4b_impl(lr, function1, 1)
    ex4b_impl(lr, function1, 1, linear_search="dichotomy")

    lr = 0.005
    ex4b_impl(lr, function2, 2)
    ex4b_impl(lr, function2, 2, linear_search="dichotomy")


def ex4c():
    def ex4c_impl(lr, function, function_number, tests_amount):
        with open(f"ex4data/ex4c/f{function_number}.txt", "w") as file:
            for i in range(tests_amount):
                start_point = np.array([random.randint(-100, 100), random.randint(-100, 100)])
                file.write(str(start_point) + '\n')
                iter_num = number_of_iterations_to_converge(lr, function, start_point)
                gradient_descent(lr, iter_num, start_point, function, draw=True,
                                 save_image=f"ex4data/ex4c/f{function_number}lr{i}.png")
                file.write("lr ")
                file.write(str(iter_num))
                file.write("\n")

                iter_num = number_of_iterations_to_converge(lr, function, start_point, linear_search="dichotomy")
                gradient_descent(lr, iter_num, start_point, function, linear_search="dichotomy", draw=True,
                                 save_image=f"ex4data/ex4c/f{function_number}d{i}.png")
                file.write("dichotomy ")
                file.write(str(iter_num))
                file.write("\n")

    ex4c_impl(0.06, function1, 1, 4)
    ex4c_impl(0.009, function2, 2, 4)


if __name__ == '__main__':
    ex4c()
