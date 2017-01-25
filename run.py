import numpy as np
import matplotlib.pyplot as plt


N = 100  # number of steps
M = 2  # dimensionality


class Cost(object):

    def __init__(self, y0, a, b, c):
        self.y0 = y0
        self.a = a
        self.b = b
        self.c = c

    def reading_speed(self, x):
        return self.y0 - self._sigmoid(x)

    def _sigmoid(self, x):
        return np.divide(self.a, 1 + np.exp(self.c * (self.b - x)))

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
        deriv_x1 = 2 * (self.y0 - sig - x[0]) * sig * (sig / self.a - 1) * self.c
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
    cost_func = Cost(y0=10, a=3, b=500, c=0.01)
    model = Model(
        x0=np.array([10., 0.]),
        A=np.array([[1., 0.], [1., 1.]]),
        b=np.array([1., 0.]),
        cost=cost_func)
    U = np.zeros(N)
    model.forward(U)
    plt.plot(model.X[:, 0], c="b", label="uncontrolled app's speed")
    reading_speed = cost_func.reading_speed(model.X[:, 1])
    plt.plot(reading_speed, c="g", label="uncontrolled reader's speed")
    U = model.find_optimal_input(U, 10000, learning_rate=1e-4)
    model.forward(U)
    plt.plot(model.X[:, 0], c="b", linestyle="dashed", lw=2., label="controlled app's speed")
    plt.plot(cost_func.reading_speed(model.X[:, 1]), c="g", linestyle="dotted", lw=5., label="controlled reader's speed")
    plt.ylim(6, 11)
    plt.xlabel("time (second)")
    plt.ylabel("speed (word / second)")
    plt.legend(loc="best")
    plt.savefig("result.eps")


if __name__ == '__main__':
    main()
