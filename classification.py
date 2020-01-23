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
    NUM_BUCKETS = 2

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

    # Calculate entropy for samples
    def h(self, samples):
        labels = set(samples[:,-1])
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

    # Return best node to split on
    def find_best_node(self, dataset):
        num_features = len(dataset[0]) - 1
        best_info_gain = 0
        best_feature = -1

        # Iterate through every attribute in dataset
        for i in num_features:
            buckets = self.find_best_split(dataset, i)
            info_gain = self.info_gain(dataset, buckets, len(dataset))
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature

    # HELPPPPPP
    def find_best_split(self, dataset, column):
        parent_entropy = self.h(dataset)

        sorted_col = set(dataset[:, column])

        possible_splits = list(itertools.combinations(sorted_col, self.NUM_BUCKETS - 1))

        for split in possible_splits:
            split = sorted(list(split))
            buckets = self.perform_split(dataset, column, split)
             # calculate IG

    def perform_split(self, data, column, split):
        buckets = []

        # First bucket (upper bounded)
        buckets.append([i for i in data[column] if i <= split[0]])

        # Intermediate buckets (upper and lower bounded)
        for index in range(1, len(split)):
            lower_bound = split[index - 1]
            upper_bound = split[index]
            buckets.append([i for i in data[column] if lower_bound < i <= upper_bound])

        # Last bucket (lower bound)
        buckets.append([i for i in data[column] if i > split[len(split) - 1]])

        return buckets

if __name__ == "__main__":
    x = DecisionTreeClassifier()
    print(x.perform_split([list(range(10))], 0, [3,6]))











        

