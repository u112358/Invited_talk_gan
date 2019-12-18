import numpy as np
import scipy.io as sio


class Dataset:
    def __init__(self, seed=2019, size=1000):
        np.random.seed(seed)
        self.size = size
        # self.data, self.label = self.generate_gmm()
        self.data = self.generate_circle()
        sio.savemat('data.mat', {'data': self.data})
        self.current_index = 0

    def generate_circle(self, center=None, r=None, sigma=None):
        if sigma is None:
            sigma = [[0.01, 0], [0, 0.01]]
        if center is None:
            center = [0, 0]
        if r is None:
            r = 0.5
        data = np.zeros((self.size, 2))
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi, self.size)
        x = np.sin(theta) + center[0]
        y = np.cos(theta) + center[1]
        noise = np.random.multivariate_normal([0, 0], sigma, size=self.size)
        data[:, 0] = x + noise[:, 0]
        data[:, 1] = y + noise[:, 1]
        return data

    def generate_gmm(self, c=3, mu=None, sigma=None):
        if sigma is None:
            # sigma = [[[0.01, 0], [0, 0.01]], [[0.01, 0], [0, 0.01]], [[0.01, 0], [0, 0.01]]]
            sigma = [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]

        if mu is None:
            # mu = [[0.1, 0.1], [0.9, 0.1], [0.4, 0.9]]
            mu = [[-10, 4], [3, 9], [6, 0]]
        data = np.empty((self.size * c, 2))
        label = np.empty(self.size * c)
        for i in range(c):
            _mu = mu[i]
            _sigma = sigma[i]
            data[self.size * i:self.size * (i + 1), :] = np.random.multivariate_normal(mean=_mu, cov=_sigma,
                                                                                       size=self.size)
            label[self.size * i:self.size * (i + 1)] = np.ones(self.size) * i
        self.size = self.size * 3
        return data, label

    def next_batch(self, batch_size=50):
        if self.current_index + batch_size > self.size:
            self.current_index = 0
        data = self.data[self.current_index:self.current_index + batch_size, :]
        self.current_index += batch_size
        return data
