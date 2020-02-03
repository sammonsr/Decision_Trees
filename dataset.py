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

if __name__ == "__main__":
    print(Dataset().load_data('toy.txt'))