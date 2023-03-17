from typing import Callable
import numpy as np


class Function:
    __calc_impl: Callable[[np.ndarray], float]
    __grad_impl: Callable[[np.ndarray], np.ndarray]

    function_calc_amount: int = 0
    grad_calc_amount: int = 0

    def __init__(self, fun: Callable, grad: Callable):
        self.__calc_impl = fun
        self.__grad_impl = grad

    def calc(self, point: np.ndarray) -> float:
        self.function_calc_amount += 1
        return self.__calc_impl(point)

    def grad(self, point: np.ndarray) -> np.ndarray:
        self.grad_calc_amount += 1
        return self.__grad_impl(point)

    def reset_calc_amount(self):
        self.function_calc_amount = 0
        self.grad_calc_amount = 0
