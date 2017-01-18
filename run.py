import numpy as np


N = 10  # number of steps
M = 2  # dimensionality


class Cost(object):

    def __init__(self, y0, a, b):
        self.y0 = y0
        self.a = a
        self.b = b

    def _sigmoid(self, x):
        return np.divide(self.a, 1 + np.exp(self.b - x))

    def __call__(self, x):
        return np.sum(
            np.square(
                x[:, 0] - self.y0 + self._sigmoid(x[:, 1])
            )
        )

    def derivative(self, x):
        assert x.ndim == 1
        sig = self._sigmoid(x[1])
        deriv_x0 = 2 * (x[0] - self.y0 + sig)
        deriv_x1 = 2 * (self.y0 - sig - x[0]) * sig * (sig - 1)
        deriv = np.array([deriv_x0, deriv_x1])
        return deriv


class Model(object):

    def __init__(self, x0, A, b, cost):
        """
        define system model
        x_{t+1} = Ax_t + bu

        Parameters
        ----------
        x0 : (M,) ndarray
        A : (M, M) ndarray
        b : (M,) ndarray
        """
        self.x0 = x0
        self.A = A
        self.b = b
        self.cost = cost

    def forward(self, U):
        X = [self.x0]
        for u in U:
            X.append(np.dot(self.A, X[-1]) + self.b * u)
        self.X = np.asarray(X)

    def backward(self, U):
        lambda_N = self.cost.derivative(self.X[-1])
        lambda_list = [lambda_N]
        for u, x in zip(U[::-1], self.X[-2::-1]):
            lambda_list.insert(
                0, self.cost.derivative(x) + np.dot(self.A.T, lambda_list[0]))
        self.lambdas = np.asarray(lambda_list)

    def update(self, U, learning_rate):
        delta_u = []
        for lambda_ in self.lambdas[:-1]:
            delta_u.append(np.dot(self.b, lambda_))
        delta_u = np.asarray(delta_u)
        return U - learning_rate * delta_u

    def find_optimal_input(self, initialU, n_iter, learning_rate):
        U = initialU
        for i in range(n_iter):
            self.forward(U)
            self.backward(U)
            U = self.update(U, learning_rate)
        return U


def main():
    model = Model(
        x0=np.array([1., 0.]),
        A=np.array([[1., 0.], [1., 1.]]),
        b=np.array([1., 0.]),
        cost=Cost(y0=2, a=1, b=-5))
    U = np.zeros(N)
    model.forward(U)
    print(model.cost(model.X))
    U = model.find_optimal_input(U, 10, learning_rate=1e-5)
    print(U)
    model.forward(U)
    print(model.cost(model.X))


if __name__ == '__main__':
    main()
