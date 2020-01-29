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
from visualiser import Visualiser


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


def create_tree():
    return Intermediate([], -1)


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
    MAX_BUCKETS = 3

    def __init__(self):
        self.is_trained = False
        self.decision_tree = None
        self.label_dict = {}

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

        reverse_dict = {}
        ordered_labels = list(set(y))
        for i in range(len(ordered_labels)):
            self.label_dict[i] = ordered_labels[i]
            # Set inputs for training as int values (to then lookup at the end)
            reverse_dict[ordered_labels[i]] = i

        y = [reverse_dict[label] for label in y]

        dataset = np.concatenate((x, np.array([y]).T), axis=1)
        self.decision_tree = create_tree()
        self.build_tree(dataset, self.decision_tree)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        vis = Visualiser(self.decision_tree)
        vis.create_plot()

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
        # predictions = np.zeros((x.shape[0],), dtype=np.object)

        root = self.decision_tree
        return [self.label_dict[self.traverse_tree(row, root)] for row in x]

    def traverse_tree(self, row, current_node):
        if type(current_node) is Leaf:
            return current_node.value

        value = row[current_node.attr_index]

        for i in range(len(current_node.children)):
            child = current_node.children[i]
            if current_node.branch_conditions[i](value):
                return self.traverse_tree(row, child)

    def build_tree(self, dataset, root):
        best_column = self.find_best_attribute(dataset)
        best_partitioning = self.find_best_partitioning(dataset, best_column)
        print("145", len(best_partitioning))
        children_datasets = self.perform_partitioning(dataset, best_column, best_partitioning)

        for i in range(len(children_datasets)):
            child_dataset = children_datasets[i]

            # do all samples have same label - Leaf case
            class_index = len(dataset[0]) - 1
            all_same_class = all([d[class_index] == child_dataset[0][class_index] for d in child_dataset])

            if all_same_class:
                class_value = child_dataset[0][class_index]
                root.add_child(Leaf(class_value), best_partitioning[i])
                continue

            # Non-Leaf case
            intermediate = Intermediate([], i)
            root.add_child(intermediate, best_partitioning[i])

            self.build_tree(child_dataset, intermediate)

    # Calculate entropy for samples
    def h(self, samples):
        labels = set([a[-1] for a in samples])
        sum = 0

        for label in labels:
            rows = [a for a in labels if a == label]
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
            partitioning = self.find_best_partitioning(dataset, i)
            print("****************")
            print("200", len(partitioning))
            print(dataset)
            buckets = self.perform_partitioning(dataset, i, partitioning)
            info_gain = self.info_gain(self.h(dataset), buckets, len(dataset))
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    def calc_num_buckets(self, dataset):
        return min(len(dataset), self.MAX_BUCKETS)

    def find_best_partitioning(self, dataset, column):
        assert column < len(dataset[0])

        num_buckets = self.calc_num_buckets(dataset)

        parent_entropy = self.h(dataset)

        # We want a list not a set? np.distinct?
        sorted_col = set([a[column] for a in dataset])

        possible_partitioning_bounds = list(itertools.combinations(sorted_col, num_buckets - 1))

        possible_partitionings = [generate_partitioning(bounds) for bounds in possible_partitioning_bounds]

        best_ig = -1
        best_partitioning = []

        for partitioning in possible_partitionings:
            print("225", len(partitioning))
            buckets = self.perform_partitioning(dataset, column, partitioning)

            ig = self.info_gain(parent_entropy, buckets, len(dataset))

            if ig > best_ig:
                best_ig = ig
                best_partitioning = partitioning

        return best_partitioning

    def perform_partitioning(self, dataset, column, partitioning):
        assert 0 < len(partitioning) <= self.MAX_BUCKETS

        buckets = [[] for _ in range(len(partitioning))]

        for i in range(len(dataset)):
            for j in range(len(partitioning)):
                in_partition = partitioning[j]
                if in_partition(dataset[i][column]):
                    buckets[j].append(dataset[i])

        return buckets
