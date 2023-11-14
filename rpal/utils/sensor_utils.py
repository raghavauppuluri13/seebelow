import numpy as np


class RingBuffer:
    """Ring buffer for debouncing sensor values"""

    def __init__(self, capacity, stability_threshold=0.01):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=float)
        self.index = 0
        self.full = False
        self.stability_threshold = stability_threshold

    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def get(self):
        if self.full:
            return np.roll(self.buffer, -self.index)
        return self.buffer[: self.index]

    @property
    def is_stable(self):
        if self.full:
            return np.std(self.buffer) < self.stability_threshold
        return False

    def __str__(self):
        return str(self.get())
