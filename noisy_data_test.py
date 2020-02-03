import numpy as np

noisy = np.genfromtxt('data/train_noisy.txt', dtype=None, delimiter=',')
full = np.genfromtxt('data/train_full.txt', dtype=None, delimiter=',')

assert noisy.shape == full.shape

# Sort them as the attributes are the same (reordered) in both.
noisy.sort()
full.sort()

with_same_label = 0

for i in range(len(full)):
    noisy_row = list(noisy[i])
    full_row = list(full[i])
    noisy_attrs = noisy_row[:-1]
    noisy_label = noisy_row[-1]
    full_attrs = full_row[:-1]
    full_label = full_row[-1]

    # This should be the case due to the sort
    assert noisy_attrs == full_attrs

    if noisy_label == full_label:
        with_same_label += 1

print("Proportion of labels different in noisy", 1 - (with_same_label / len(full)))
