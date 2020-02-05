import random

import numpy as np

class Dataset:

    def load_data(self, filename):
        data = np.genfromtxt('data/' + filename, dtype=None, delimiter=',')

        x = []
        y = []

        for row in data:
            x.append(list(row)[:-1])
            y.append(list(row)[-1])

        x = np.array(x)
        y = np.array(y)

        return x, y

    def load_split_data(self, filename, k, seed=None):
        xs, ys = self.load_data(filename)
        random.seed(seed)

        subsets = []
        for i in range(k):
            subsets.append(([], []))

        shuffled_order = list(range(len(xs)))
        random.shuffle(shuffled_order)

        for i in range(len(shuffled_order)):
            chosen_index = shuffled_order[i]
            subset_index = i % k

            subsets[subset_index][0].append(xs[chosen_index])
            subsets[subset_index][1].append(ys[chosen_index])

        return subsets


if __name__ == "__main__":
    print(Dataset().load_data('toy.txt'))