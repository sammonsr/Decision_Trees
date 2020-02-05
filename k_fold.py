from classification import DecisionTreeClassifier
from dataset import Dataset
from eval import Evaluator
import numpy as np


# TODO: Fix this!
def perform_k_fold(k, classifier_class, filename, random_seed=None, testing=False):
    subsets = Dataset().load_split_data(filename, k, random_seed)
    evaluator = Evaluator()
    accuracies = []

    best_model = None

    for i in range(k):
        classifier = classifier_class()

        # Select ith subset as test set
        test_set = subsets[i]

        # Select set as validation set
        validation_set = subsets[(i + 1) % k]

        comparison_set = test_set if testing else validation_set

        # Select all others as training set
        training_sets = [subsets[j] for j in range(len(subsets)) if j not in [i, (i + 1) % k]]

        # run classifier on the above

        training_set_x = []
        training_set_y = []
        for training_set in training_sets:
            training_set_x.extend(training_set[0])
            training_set_y.extend(training_set[1])

        classifier.train(np.array(training_set_x), np.array(training_set_y))

        predictions = classifier.predict(comparison_set[0])

        confusion = evaluator.confusion_matrix(predictions, comparison_set[1])

        # record score
        accuracy = evaluator.accuracy(confusion)
        if len(accuracies) != 0 and accuracy > max(accuracies):
            best_model = classifier

        accuracies.append(accuracy)

    accuracies = np.array(accuracies)

    return accuracies.sum() / k, np.std(accuracies), best_model


def print_metrics(confusion):
    print("macro_p = ", evaluator.precision(confusion)[1])
    print("macro_r = ", evaluator.recall(confusion)[1])
    print("macro_f = ", evaluator.f1_score(confusion)[1])
    print("accuracy = ", evaluator.accuracy(confusion))


if __name__ == "__main__":
    dataset = Dataset()
    evaluator = Evaluator()
    # Load test.txt for evaluating both models
    x_test, y_test = dataset.load_data('test.txt')

    # Train on full dataset
    x, y = dataset.load_data('train_full.txt')
    classifier_classic = DecisionTreeClassifier()
    classifier_classic = classifier_classic.train(x, y)
    predictions_classic = classifier_classic.predict(x_test)

    avg_acc, sd, classifier_kfold = perform_k_fold(10, DecisionTreeClassifier, 'train_full.txt', random_seed=500,
                                                   testing=False)
    predictions_kfold = classifier_kfold.predict(x_test)

    # Test both models on test.txt
    confusion_classic = evaluator.confusion_matrix(predictions_classic, y_test)
    confusion_kfold = evaluator.confusion_matrix(predictions_kfold, y_test)

    # Compare metrics
    print("Classic classifier:")
    print_metrics(confusion_classic)
    print()
    print("k-fold classifier:")
    print_metrics(confusion_kfold)
