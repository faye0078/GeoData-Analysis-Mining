import numpy as np

class minValue():
    def __init__(self, min_num):
        self.min_values = np.zeros(min_num)
        self.min_indexes= np.array([None]*min_num)

    def update(self, n_value, n_index):
        for i, value in enumerate(self.min_values):
            if n_value <= value:
                if i == len(self.min_indexes) -1 or n_value >= self.min_values[i + 1]:
                    self.min_values[i] = n_value
                    self.min_indexes[i] = n_index
                    break
            else:
                continue

    def get_values(self):
        return self.min_values