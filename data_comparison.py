from dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset()
    full = dataset.load_data('train_full.txt')
    noisy = dataset.load_data('train_noisy.txt')

    # Question 1.3.1
        # hmmmm...

    # Question 1.3.2
    all_labels = set([row[-1] for row in full])

    class_dist_noisy = {}
    class_dist_full = {}
    for label in all_labels:
        num_rows_in_full = len([0 for row in full if row[-1] == label])
        num_rows_in_noisy = len([0 for row in noisy if row[-1] == label])

        class_dist_noisy[label] = int((num_rows_in_noisy / len(noisy)) * 100)
        class_dist_full[label] = int((num_rows_in_full / len(full)) * 100)

    print(class_dist_full)
    print(class_dist_noisy)


