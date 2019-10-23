#############################################################################
# This file contains a series of utility functions for computing
# the accuracy of our machine learning models in regards to the specific
# problem.
#############################################################################

import numpy as np

def get_accuracy_A1(confusion_matrix, total_samples):
    """
    Computes the A1 accuracy given the confusion matrix and the total number
    of samples.

    The A1 accuracy is defined as the perecentage of samples where the valid track
    was correctly detected.
    
    Args:
        confusion_matrix: The confusion matrix
        total_samples (int): The total number of samples
    
    Returns:
        The A1 accuracy
    """

    return confusion_matrix[1,1] / total_samples

def get_accuracy_Ac(confusion_matrix, predictions, segmented_test_labels):
    """
    Computes the Ac accuracy given the confusion matrix, the predictions from
    the machine learning model, and the test dataset labels in segmented format.

    The Ac accuracy is defined as the percentage of predicted valid samples that contain
    mispredicted rows.
    
    Args:
        confusion_matrix: The confusion matrix
        predictions: Predictions outputted by an ML model
        segmented_test_labels: Segmented array of the test labels
    
    Returns:
        The Ac accuracy
    """

    predicted_samples_with_errors = 0

    iter_index = 0

    for sample in segmented_test_labels:
        first_label_index = iter_index
        found_invalid = False

        if predictions[first_label_index] != 1:
            iter_index += sample.size
            continue

        for label in np.nditer(sample):
            if iter_index == first_label_index:
                iter_index += 1
                continue

            if not found_invalid and predictions[iter_index] == 1:
                predicted_samples_with_errors += 1
                found_invalid = True

            iter_index += 1

    return predicted_samples_with_errors / confusion_matrix[1,1]

def get_accuracy_Ac_g(confusion_matrix, total_rows, total_samples):
    """
    Computes the Ac accuracy given the confusion matrix and the total number
    of samples.

    The Ac accuracy is defined as the global percentage of rows that were mispredicted
    as valid ones.
    
    Args:
        confusion_matrix: The confusion matrix
        total_rows (int): The total number of rows in the dataset
        total_samples (int): The total number of samples in the dataset
    
    Returns:
        The Ac_g accuracy
    """

    return confusion_matrix[0,1] / (total_rows - total_samples)

def get_accuracy_Ah(predicted_probabilities, segmented_test_labels):
    """
    Computes the Ah accuracy given an array of the predicted probabilities
    for each row in the dataset as well as the test labels in segmented
    format (using svm_utils).

    The Ah accuracy is defined as the perecentage of predictions where the
    label correctly predicted as 1 has the highest probability of all other labels
    being a 1 in a sample.
    
    Args:
        predicted_probabilities: Flat array of predicted probabilities in tuple format
        (0: probability of being 0, 1: probability of being 1)
        segmented_test_labels: Segmented array of the test labels
    
    Returns:
        The Ah accuracy
    """

    correct_predictions = 0

    max_index = 0
    iter_index = 0

    for sample in segmented_test_labels:
        first_label_index = iter_index
        max_index = first_label_index

        for label in np.nditer(sample):
            if predicted_probabilities[iter_index][1] > predicted_probabilities[max_index][1]:
                max_index = iter_index

            iter_index += 1

        # If the highest probability of 1 has the index of the first label
        # in the sample, then it is correct
        if max_index == first_label_index:
            correct_predictions += 1

    total_samples = len(segmented_test_labels)

    return correct_predictions / total_samples

def get_accuracy_Af(confusion_matrix, total_samples):
    """
    Computes the Af accuracy given the confusion matrix and the total number
    of samples.

    The Af accuracy is defined as the perecentage of samples where the valid track
    was not detected.
    
    Args:
        confusion_matrix: The confusion matrix
        total_samples (int): The total number of samples
    
    Returns:
        The Af accuracy
    """

    return confusion_matrix[1,0] / total_samples