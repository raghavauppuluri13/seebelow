import numpy as np


class RunningStats:
    """from https://github.com/eanswer/TactileSimulation/blob/main/utils/running_mean_std.py"""

    def __init__(self, shape=()):
        self._mean = np.zeros(shape, "float64")
        self._var = np.ones(shape, "float64")
        self._count = 0

    def update(self, values):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]

        delta = batch_mean - self._mean
        tot_count = self._count + batch_count

        new_mean = self._mean + delta * batch_count / tot_count
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self._count * batch_count / (self._count + batch_count)
        )
        new_var = m_2 / (self._count + batch_count)
        new_count = batch_count + self._count

        self._mean = new_mean
        self._var = new_var
        self._count = new_count

    def normalize(self, value):
        return (value - self._mean) / np.sqrt(self._var + 1e-5)


class RingBuffer:
    """Ring buffer for normalizing and debouncing sensor values"""

    def __init__(self, capacity, buffer=None, dtype=float):
        if buffer is None:
            self._capacity = capacity
            self._buffer = np.empty(capacity, dtype=dtype)
        else:
            self._buffer = buffer
            self._capacity = len(self._buffer)
        self._index = 0
        self._initialized = False

    def append(self, value):
        self._buffer[self._index] = value
        self._index = (self._index + 1) % self._capacity
        if self._index == 0:
            self._initialized = True

    def get(self):
        return self._buffer[: self._index]

    def overflowed(self):
        return self._index == 0 and self._initialized

    @property
    def std(self):
        return np.std(self._buffer)

    @property
    def buffer(self):
        return self._buffer

    def __str__(self):
        return str(self.get())
