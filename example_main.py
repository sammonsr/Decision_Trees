##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

import dataset
from classification import DecisionTreeClassifier
from eval import Evaluator

if __name__ == "__main__":
    print("Loading the training dataset...");

    dataset = dataset.Dataset()

    x, y = dataset.load_data('train_full.txt')


    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(x, y, show_diagram=True)

    print("Loading the test set...")

    x_test, y_test = dataset.load_data('test.txt')

    
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    
    classes = sorted(list(set(y_test)))
    
    print("Evaluating test predictions...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)
    
    print("Confusion matrix:")
    print(confusion)

    accuracy = evaluator.accuracy(confusion)
    print()
    print("Accuracy: {}".format(accuracy))

    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));
   
    print() 
    print("Macro-averaged Precision: {:.2f}".format(macro_p))
    print("Macro-averaged Recall: {:.2f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))

