
import numpy as np
class Sgd:
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor, flag=False):
        update_weight = weight_tensor - self.learning_rate * gradient_tensor
        return update_weight

    def calculate_update_bias(self, weight_tensor, gradient_tensor, flag=False):
        # if it is the first iteration, then create self.velocity with the same shape of weight_tensor

        update_weight = weight_tensor - self.learning_rate * gradient_tensor
        return update_weight




class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor, flag=False):
        # if it is the first iteration, then create self.velocity with the same shape of weight_tensor

        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.velocity

        return updated_weights

    def calculate_update_bias(self, weight_tensor, gradient_tensor, flag=False):
        # if it is the first iteration, then create self.velocity with the same shape of weight_tensor

        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.velocity

        return updated_weights


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.v = 0
        self.r = 0

        self.v_b = 0
        self.r_b = 0
        self.k = 1
        self.t = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor ** 2

        v_ = self.v / (1 - self.mu ** self.k)
        r_ = self.r / (1 - self.rho ** self.k)

        self.k += 1

        return weight_tensor - self.learning_rate * v_ / (np.sqrt(r_) + self.epsilon)

    def calculate_update_bias(self, weight_tensor, gradient_tensor):
        self.v_b = self.mu * self.v_b + (1 - self.mu) * gradient_tensor
        self.r_b = self.rho * self.r_b + (1 - self.rho) * gradient_tensor ** 2

        v_ = self.v_b / (1 - self.mu ** self.t)
        r_ = self.r_b / (1 - self.rho ** self.t)

        self.t += 1

        return weight_tensor - self.learning_rate * v_ / (np.sqrt(r_) + self.epsilon)
