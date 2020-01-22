import numpy as np

class Dataset:

    def load_data(self, filename):
        data = np.genfromtxt('data/' + filename, dtype=None, delimiter=',')
        return data

if __name__ == "__main__":
    print(Dataset().load_data('toy.txt'))