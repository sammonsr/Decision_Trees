##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
##############################################################################
import itertools
import math

import numpy as np

from tree import Leaf, Intermediate


def generate_partitioning(bounds):
    partitioning = []

    # First partition (upper bounded)
    partitioning.append(lambda a: a < bounds[0])

    # Intermediate partitions (upper and lower bounded)
    for i in range(1, len(bounds)):
        if i == len(bounds) - 1:
            partitioning.append(lambda a: bounds[i - 1] <= a < bounds[i])

    # Last partition (lower bounded)
    partitioning.append(lambda a: a >= bounds[len(bounds) - 1])

    return partitioning


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """
    NUM_BUCKETS = 3

    def __init__(self):
        self.is_trained = False

    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # remember to change this if you rename the variable
        return predictions

    def build_tree(self, dataset, root=None):
        if root is None:
            root = Intermediate(None, [])

        # do all samples have same label
        class_index = len(dataset[0]) - 1
        all_same_class = all([d[class_index] == dataset[0][class_index] for d in dataset])

        if all_same_class:
            class_value = dataset[0][class_index]
            return Leaf(class_value)

        best_column = self.find_best_attribute(dataset)
        best_partitioning = self.find_best_partitioning(dataset, best_column)
        children_datasets = self.perform_partitioning(dataset, best_column, best_partitioning)

        for i in range(len(children_datasets)):
            child_dataset = children_datasets[i]

            intermediate = Intermediate(best_partitioning[i], [])
            intermediate.add_child(self.build_tree(child_dataset, intermediate))
            root.add_child(intermediate)

        return root

    # Calculate entropy for samples
    def h(self, samples):
        labels = set(samples[:, -1])
        sum = 0

        for label in labels:
            rows = [a for a in labels if a[-1] == label]
            pc = len(rows) / len(samples)
            sum -= pc * math.log2(pc)

        return sum

    # Calculate weighted average of set of samples
    def h_prime(self, sets, num_samples):
        sum = 0

        for s in sets:
            sum += (len(s) / num_samples) * self.h(s)

        return sum

    # Calculate information gain
    def info_gain(self, parent_entropy, children, num_samples):
        return parent_entropy - self.h_prime(children, num_samples)

    # Return best node to partition on
    def find_best_attribute(self, dataset):
        num_features = len(dataset[0]) - 1
        best_info_gain = 0
        best_feature = -1

        # Iterate through every attribute in dataset
        for i in range(num_features):
            buckets = self.find_best_partitioning(dataset, i)
            info_gain = self.info_gain(dataset, buckets, len(dataset))
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    def find_best_partitioning(self, dataset, column):
        assert column < len(dataset[0])

        parent_entropy = self.h(dataset)

        sorted_col = set(dataset[:, column])

        possible_partitioning_bounds = list(itertools.combinations(sorted_col, self.NUM_BUCKETS - 1))

        possible_partitionings = [generate_partitioning(bounds) for bounds in possible_partitioning_bounds]


        best_ig = -1
        best_partitioning = []

        for partitioning in possible_partitionings:
            buckets = self.perform_partitioning(dataset, column, partitioning)

            ig = self.info_gain(parent_entropy, buckets, len(dataset))

            if ig > best_ig:
                best_ig = ig
                best_partitioning = partitioning

        return best_partitioning

    def perform_partitioning(self, dataset, column, partitioning):
        assert self.NUM_BUCKETS == len(partitioning)

        buckets = [[] for _ in range(self.NUM_BUCKETS)]

        for i in range(len(dataset)):
            for j in range(len(partitioning)):
                in_partition = partitioning[j]
                if in_partition(dataset[i][column]):
                    buckets[j].append(dataset[i])

        return buckets


if __name__ == "__main__":
    x = DecisionTreeClassifier()
    data = [[i] for i in range(10)]
    print(data)
    print()
    partitioning = generate_partitioning([4, 7])
    print(x.perform_partitioning(data, 0, partitioning))
