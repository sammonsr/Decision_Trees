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

from condition import Condition
from dataset import Dataset
from eval import Evaluator
from tree import Leaf, Intermediate


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
    DEFAULT_BUCKETS = 2

    def __init__(self):
        self.is_trained = False
        self.decision_tree = None
        self.label_dict = {}

    def train(self, x, y, show_diagram=False):
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

        #self.prune(Dataset().load_data('validation.txt'))

        if show_diagram:
            print("The maximum depth is: ", self.get_max_depth())
            print(self.decision_tree.__str__(label_dict=self.label_dict))

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
        return np.array([self.label_dict[self.traverse_tree(row, root)] for row in x])

    def traverse_tree(self, row, current_node):
        if type(current_node) is Leaf:
            return current_node.value

        value = row[current_node.attr_index]

        for i in range(len(current_node.children)):
            child = current_node.children[i]
            if current_node.branch_conditions[i].condition_lambda(value):
                return self.traverse_tree(row, child)

    # If one of the buckets is empty, convert node into a leaf
    def parent_to_leaf(self, root, dataset):
        assert root.parent is not None, "Cannot prune the root"

        parent = root.parent
        label_votes = {}

        max_num_votes = -1
        most_popular = None

        for row in dataset:
            label = row[-1]
            if label not in label_votes.keys():
                label_votes[label] = 1
            else:
                label_votes[label] = label_votes[label] + 1
            if label_votes[label] > max_num_votes:
                max_num_votes = label_votes[label]
                most_popular = label

        parent.replace_child(root.index_in_parent, Leaf(most_popular))

    def check_accuracy(self, validation_dataset):
        x_test, y_test = validation_dataset

        predictions = self.predict(x_test)
        evaluator = Evaluator()
        return evaluator.accuracy(evaluator.confusion_matrix(predictions, y_test))

    def build_tree(self, dataset, root):
        best_column = self.find_best_attribute(dataset)
        best_partitioning = self.find_best_partitioning(dataset, best_column)

        # If we can't partition then convert parent to leaf
        if len(best_partitioning) == 0:
            self.parent_to_leaf(root, dataset)
            return

        children_datasets = self.perform_partitioning(dataset, best_column, best_partitioning)

        # Edge case: Ensure root node has attr_index, entropy and class dist set
        root.attr_index = best_column
        root.entropy = self.h(dataset)
        root.class_dist = self.get_class_dist(dataset)

        for i in range(len(children_datasets)):
            child_dataset = children_datasets[i]

            # do all samples have same label - Leaf case
            class_index = len(dataset[0]) - 1
            all_same_class = all([d[class_index] == child_dataset[0][class_index] for d in child_dataset])

            if len(child_dataset) == 0:
                # Not enough data to split on parent so prune
                self.parent_to_leaf(root, dataset)
                return
            elif all_same_class:
                class_value = child_dataset[0][class_index]
                root.add_child(Leaf(class_value), best_partitioning[i])
                continue

            # Non-Leaf case
            intermediate = Intermediate([], best_column)
            intermediate.entropy = self.h(child_dataset)
            intermediate.class_dist = self.get_class_dist(child_dataset)
            root.add_child(intermediate, best_partitioning[i])

            self.build_tree(child_dataset, intermediate)

    def get_class_dist(self, dataset):
        dist = {}
        uniq_labels = set(map(lambda v: self.label_dict[v], [row[-1] for row in dataset]))

        for label in uniq_labels:
            dist[label] = len([a for a in dataset if self.label_dict[a[-1]] == label])
        return dist

    # Calculate entropy for samples
    def h(self, samples):
        labels = set([a[-1] for a in samples])
        sum = 0

        for label in labels:
            rows = [s for s in samples if s[-1] == label]
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
            if len(partitioning) == 0:
                continue
            buckets = self.perform_partitioning(dataset, i, partitioning)
            info_gain = self.info_gain(self.h(dataset), buckets, len(dataset))
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    def calc_num_buckets(self, dataset):
        return min(len(dataset), self.DEFAULT_BUCKETS)

    def generate_partitioning(self, bounds):
        partitioning = []

        # First partition (upper bounded)
        partitioning.append(Condition(lambda a: a < bounds[0], "x < {}".format(bounds[0])))

        # Intermediate partitions (upper and lower bounded)
        for i in range(1, len(bounds)):
            if i == len(bounds) - 1:
                partitioning.append(Condition(lambda a: bounds[i - 1] <= a < bounds[i],
                                              "{} <= x < {}".format(bounds[i - 1], bounds[i])))

        # Last partition (lower bounded)
        partitioning.append(
            Condition(lambda a: a >= bounds[len(bounds) - 1], "x >= {}".format(bounds[len(bounds) - 1])))

        return partitioning

    def find_best_partitioning(self, dataset, column):
        assert column < len(dataset[0])

        num_buckets = self.calc_num_buckets(dataset)

        parent_entropy = self.h(dataset)

        # We want a list not a set? np.distinct?
        sorted_col = set([a[column] for a in dataset])

        possible_partitioning_bounds = list(itertools.combinations(sorted_col, num_buckets - 1)) + list(
            itertools.combinations(sorted_col, num_buckets))

        possible_partitionings = [self.generate_partitioning(bounds) for bounds in possible_partitioning_bounds]

        best_ig = -1
        best_partitioning = []

        for partitioning in possible_partitionings:
            if not 0 < len(partitioning) <= self.DEFAULT_BUCKETS:
                pass
            buckets = self.perform_partitioning(dataset, column, partitioning)

            ig = self.info_gain(parent_entropy, buckets, len(dataset))

            if ig > best_ig:
                best_ig = ig
                best_partitioning = partitioning

        return best_partitioning

    def perform_partitioning(self, dataset, column, partitioning):
        buckets = [[] for _ in range(len(partitioning))]

        for i in range(len(dataset)):
            for j in range(len(partitioning)):
                in_partition = partitioning[j]
                if in_partition.condition_lambda(dataset[i][column]):
                    buckets[j].append(dataset[i])

        return buckets

    ''' Prune works by performing the following steps:
        1.) Find all leaf nodes at max depth
        2.) For each leaf node,
            Prune the parent using the majority vote method.
              - If the new accuracy is more than or equal to the previous accuracy, call prune again on the new tree
              - Otherwise, restore the the tree to it's state before the prune, and move onto the next leaf and repeat step 2
    '''
    def prune(self, validation_dataset):
        tree_before = self.decision_tree.__str__(label_dict=self.label_dict)

        max_depth = self.get_max_depth()
        lowest_leafs = self.get_leaf_nodes_at_depth(max_depth)

        # Check accuracy of tree
        pre_acc = self.check_accuracy(validation_dataset)

        for leaf in lowest_leafs:
            parent = leaf.parent
            grandparent = parent.parent

            # Variables used for majority vote procedure
            label_votes = {}
            max_num_votes = -1
            most_popular = None

            # Prune tree
            for i, child in enumerate(parent.children):
                value = child.value

                if value not in label_votes.keys():
                    label_votes[value] = 1
                else:
                    label_votes[value] = label_votes[value] + 1

                if label_votes[value] > max_num_votes:
                    max_num_votes = label_votes[value]
                    most_popular = value

            # Prune parent using majority vote
            grandparent.replace_child(parent.index_in_parent, Leaf(most_popular))

            # check that tree changed
            tree_after = self.decision_tree.__str__(label_dict=self.label_dict)

            assert tree_before != tree_after

            # Check accuracy of pruned tree
            post_acc = self.check_accuracy(validation_dataset)

            # If accuracy >= previous accuracy, call prune on the new tree
            # Else try and prune using next leaves
            if post_acc >= pre_acc:
                self.prune(validation_dataset)
                return
            else:
                # Restore state of tree before attempted prune, and move onto next leaf
                grandparent.replace_child(parent.index_in_parent, parent)

    def get_max_depth(self, root=None, depth=0):
        depths = []
        if root is None:
            root = self.decision_tree

        if type(root) is Leaf:
            depths.append(depth)
        else:
            for child in root.children:
                depths.append(self.get_max_depth(child, depth + 1))

        return max(depths)

    def get_leaf_nodes_at_depth(self, depth, root=None, current_depth=0):
        leafs = []

        if root is None:
            root = self.decision_tree

        if type(root) is Leaf:
            if depth == current_depth:
                return [root]
            else:
                return []

        for child in root.children:
            leafs.extend(self.get_leaf_nodes_at_depth(depth, child, current_depth + 1))

        return leafs
