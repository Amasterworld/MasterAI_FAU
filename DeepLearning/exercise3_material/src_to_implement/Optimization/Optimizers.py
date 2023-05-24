
import numpy as np

class Optimizer:
    def  __init__(self):
        self.regularizer = None
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

""""
to easy to understand: in the unittest:
        optimizer = Optimizers.Sgd(2)
        regularizer = Constraints.L1_Regularizer(2)
        # Note optimizer now is object of  Optimizers.Sgd(2)
        optimizer.add_regularizer(regularizer) 
    that mean  self.regularizer now store the object of Constraints.L1_Regularizer(2), alpha = 2
"""
class Sgd(Optimizer):

    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
            shrinked_weights = weight_tensor - self.learning_rate * regularizer_gradient
            update_weight = shrinked_weights - self.learning_rate * gradient_tensor
        else:

            update_weight = weight_tensor - self.learning_rate * gradient_tensor
        return update_weight
    def calculate_update_bias(self, weight_tensor, gradient_tensor):
        # if it is the first iteration, then create self.velocity with the same shape of weight_tensor
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
            shrinked_weights = weight_tensor - self.learning_rate * regularizer_gradient
            update_weight = shrinked_weights - self.learning_rate * gradient_tensor

            return update_weight
        else:

            update_weight = weight_tensor - self.learning_rate * gradient_tensor
            return update_weight




class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularization_gradient = self.regularizer.calculate_gradient(weight_tensor)
            # calculate the shrinked_weights
            shrinked_weights = weight_tensor - self.learning_rate * regularization_gradient
            self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
            updated_weights = shrinked_weights + self.velocity
        else:
            self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
            updated_weights = weight_tensor + self.velocity
        return updated_weights

    def calculate_update_bias(self, weight_tensor, gradient_tensor, flag=False):

        # if self.regularizer is not None then calculate regularizer_gradient from calculate_gradient method
        if self.regularizer is not None:
            regularization_gradient = self.regularizer.calculate_gradient(weight_tensor)
            # calculate the shrinked_weights
            shrinked_weights = weight_tensor - self.learning_rate * regularization_gradient
            self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
            updated_weights = shrinked_weights + self.velocity
        else:
            self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
            updated_weights = weight_tensor + self.velocity
        return updated_weights


class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
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
        if self.regularizer is not None:
            regularization_gradient = self.regularizer.calculate_gradient(weight_tensor)


            shrinked_weights = weight_tensor - self.learning_rate * regularization_gradient

            return shrinked_weights - self.learning_rate * v_ / (np.sqrt(r_) + self.epsilon)
        else:
            return weight_tensor - self.learning_rate * v_ / (np.sqrt(r_) + self.epsilon)

    def calculate_update_bias(self, weight_tensor, gradient_tensor):

        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor ** 2

        v_ = self.v / (1 - self.mu ** self.k)
        r_ = self.r / (1 - self.rho ** self.k)

        self.k += 1
        if self.regularizer is not None:
            regularization_gradient = self.regularizer.calculate_gradient(weight_tensor)

            shrinked_weights = weight_tensor - self.learning_rate * regularization_gradient

            return shrinked_weights - self.learning_rate * v_ / (np.sqrt(r_) + self.epsilon)
        else:
            return weight_tensor - self.learning_rate * v_ / (np.sqrt(r_) + self.epsilon)