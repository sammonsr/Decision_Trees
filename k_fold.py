from dataset import Dataset
from eval import Evaluator

# TODO: Fix this!
def perform_k_fold(k, classifier, filename, random_seed=None):
    subsets = Dataset().load_split_data(filename, k, random_seed)
    evaluator = Evaluator()
    total_score = 0

    for i in range(k):
        # Select ith subset as test set
        test_set = subsets[i]

        # Select set as validation set
        if (i == k - 1):
            validation_set = subsets[0]
            val_index = 0

        else :
            validation_set = subsets[i+1]
            val_index = i+1

        # Select all others as training set
        training_set = [i for k, i in enumerate(subsets) if k not in [i, val_index]]

        # run classifier on the above
        score = classifier.train(training_set[0], training_set[1])

        predictions = classifier.predict(test_set[0])

        confusion = evaluator.confusion_matrix(predictions, test_set[1])

        error_score = 1 - evaluator.accuracy(confusion)

        # record score
        total_score += error_score


    # Average score
    avg_score = total_score / k

    # Return average
    return avg_score



