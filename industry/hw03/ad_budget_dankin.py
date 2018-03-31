#coding=utf-8

from sklearn.linear_model import LinearRegression  
from scipy.optimize import minimize
from scipy.optimize import linprog
import numpy as np


class Optimizer:
    def __init__(self):
        pass

    def optimize(self, origin_budget):
        default_target = self.model.predict([origin_budget])[0]
        random_gen = np.random.RandomState(42)
        best_budget = origin_budget
        print(best_budget)
        c = [np.sum(best_budget)]
        A = [best_budget, self.model.predict([best_budget])[0]]
        b = [origin_budget * 1.05, float('Inf')]
        x0_bounds = (origin_budget * 0.95, origin_budget * 1.05)
        x1_bounds = (default_target, None)
        res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds))
        '''for _ in range(10000):
            mask = (random_gen.randint(0, 5, size=len(origin_budget)) - 1) * 0.01 + 1
            new_budget = origin_budget * mask
            if self.model.predict([new_budget])[0] >= default_target and np.sum(best_budget) > np.sum(new_budget):
                best_budget = new_budget'''
        print(res)
        return best_budget

    def fit(self, X_data, y_data):
        self.model = LinearRegression().fit(X_data, y_data)

