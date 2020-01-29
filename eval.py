import numpy as np

class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        pred_annons = list(zip(prediction, annotation))

        for i in range(len(confusion)):
            label = class_labels[i]
            for j in range(len(confusion[i])):
                pred = class_labels[j]
                pred_label = (pred, label)
                count = len([a for a in pred_annons if a == pred_label])
                confusion[i][j] = count

        return np.array(confusion)


    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        return np.trace(confusion) / np.sum(confusion)


    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        p = np.array([self.class_precision(confusion, i) for i in range(len(confusion))])
        macro_p = np.average(p)

        return (p, macro_p)

    def class_precision(self, conf, class_index):
        class_col_sum = conf[:, class_index].sum()
        return conf[class_index, class_index] / class_col_sum


    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        r = np.array([self.class_recall(confusion, i) for i in range(len(confusion))])
        macro_r = np.average(r)

        return (r, macro_r)

    def class_recall(self, conf, class_index):
        class_row_sum = conf[class_index, :].sum()
        return conf[class_index, class_index] / class_row_sum

    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        f = np.zeros((len(confusion), ))
        f = np.array([self.class_f1(confusion, i) for i in range(len(confusion))])
        macro_f = np.average(f)

        return (f, macro_f)

    def class_f1(self, conf, class_index):
        p = self.class_precision(conf, class_index)
        r = self.class_recall(conf, class_index)
        return 2 * ((p*r) / (p + r))
