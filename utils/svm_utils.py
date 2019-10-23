#############################################################################
# This file contains a series of utility functions for manipulating SVM data
# for the purpose of this project.
#############################################################################

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import math

def read_svm_file(filename, num_features):
    """
    Reads the SVM file with the passed name and returns
    the rows in dense format and the labels.
    
    Args:
        filename (string): The path of the SVM file
    
    Returns:
        dense rows, labels numpy arrays
    """

    data = load_svmlight_file(filename, num_features)
    data0Dense = data[0].todense()
    
    return data0Dense, data[1]

def read_concat_svm_files(filenames, num_features):
    """
    Reads the SVM files identified by the names in the list
    of filenames passed and return a concatenated version.
    
    Args:
        filenames (list(string)): Filenames of the SVM files to read and concatenate
    
    Returns:
        concatenated rows, concatenated labels ; as numpy arrays
    """

    concat_rows = []
    concat_labels = []

    for filename in filenames:
        temp_rows, temp_labels = read_svm_file(filename, num_features)

        concat_rows.append(temp_rows)
        concat_labels.append(temp_labels)

    concat_rows_np = np.concatenate(concat_rows)
    cocnat_labels_np = np.concatenate(concat_labels)

    return concat_rows_np, cocnat_labels_np

def get_total_samples(labels):
    """
    Gets the total number of samples based on the passed labels array.
    
    Args:
        labels (np.labels): The labels of the SVM data in a numpy array
    
    Returns:
        Total number of samples
    """

    total_samples = 0

    start_index = -1
    for current_index, val in enumerate(labels):
       if val == 1:
        if start_index == -1:
            start_index = current_index
        else:
            total_samples += 1
            start_index = current_index

    # Account for the last sample
    total_samples += 1

    return total_samples

def segment_svm_data(rows, labels):
    """
    Segments the passed SVM data (rows, labels) as per the instructions
    given by Gagik. Complexity is O(N). The rows must be in dense format.
    
    Args:
        rows (np.array): The rows of the SVM data in a numpy array
        labels (np.labels): The labels of the SVM data in a numpy array
    
    Returns:
        Segmented rows, segmented labels as numpy arrays
    """

    segmented_labels = []
    segmented_rows = []
    
    start_index = -1
    for current_index, val in enumerate(labels):
       if val == 1:
        if start_index == -1:
            start_index = current_index
        else:
            segmented_labels.append(labels[start_index:current_index])
            segmented_rows.append(rows[start_index:current_index])
            
            start_index = current_index

    # Add last segmented sample
    total_labels = labels.size
    segmented_labels.append(labels[start_index:total_labels])
    segmented_rows.append(rows[start_index:total_labels])
        
    return segmented_rows, segmented_labels

################################################## #############################
# SVM range functions
###############################################################################
def get_svm_range(rows, labels, start_index, end_index):
    """
    Given rows and labels as numpy arrays, extracts a range specified by the
    start and end indices (minus 1) and returns the range in a Pandas DataFrame.
    
    Args:
        rows (np.array): The rows of the SVM data in a numpy array
        labels (np.array): The labels of the SVM data in a numpy array
        start_index (int): The starting index of the range
        end_index (int): The ending index of the range (NOTE: actual data will be end_index - 1)
    
    Returns:
        Pandas DataFrame containing the specified SVM data range
    """

    return pd.DataFrame(rows[start_index:end_index], labels[start_index:end_index])

def get_segmented_svm_range(segmented_rows, segmented_labels, start_sample_index, end_sample_index):
    """
    Given segmented rows and labels as numpy arrays, extracts a range specified by the
    start and end indices (minus 1) and returns the range in a Pandas DataFrame.
    The indices correspond to samples, not individual rows.
    
    Args:
        segmented_rows (np.array): The rows of the segmented SVM data in a numpy array
        segmented_labels (np.array): The labels of the segmented SVM data in a numpy array
        start_sample_index (int): The starting index of the sample range
        end_sample_index (int): The ending index of the sample range (NOTE: actual data will be ending index - 1)
    
    Returns:
        Pandas DataFrame containing the specified SVM data range in unsegmented format.
    """

    # Flatten data
    flat_rows = np.vstack(segmented_rows[start_sample_index:end_sample_index])
    flat_labels = np.concatenate(segmented_labels[start_sample_index:end_sample_index]).ravel()

    return pd.DataFrame(flat_rows, flat_labels)

###############################################################################
# Sample retrieval functions
###############################################################################
def get_rows_segmented_sample_range(segmented_rows, start_sample_index, end_sample_index):
    """
    Given segmented rows as a numpy array, extracts a sample range bounded
    by the passed indices. The end of the range is ending index - 1.
    
    Args:
        segmented_rows (np.array): The rows of the segmented SVM data in a numpy array
        start_sample_index (int): The starting index of the sample range
        end_sample_index (int): The ending index of the sample range (NOTE: actual data will be ending index - 1)
    
    Returns:
        Pandas DataFrame containing the sample rows.
    """

    flat_rows = np.vstack(segmented_rows[start_sample_index:end_sample_index])
    return pd.DataFrame(flat_rows)

def get_labels_segmented_sample_range(segmented_labels, start_sample_index, end_sample_index):
    """
    Given segmented labels as a numpy array, extracts a sample range bounded
    by the passed indices. The end of the range is ending index - 1.
    
    Args:
        segmented_labels (np.array): The labels of the segmented SVM data in a numpy array
        start_sample_index (int): The starting index of the sample range
        end_sample_index (int): The ending index of the sample range (NOTE: actual data will be ending index - 1)
    
    Returns:
        Pandas DataFrame containing the sample labels.
    """

    flat_labels = np.concatenate(segmented_labels[start_sample_index:end_sample_index]).ravel()
    return pd.DataFrame(flat_labels)

def get_rows_segmented_sample(segmented_rows, index):
    """
    Given segmented rows as a numpy array, extracts the segmented sample under the given index.
    
    Args:
        segmented_rows (np.array): The rows of the segmented SVM data in a numpy array
        index (int): The index of the segmented sample rows to get
    
    Returns:
        Pandas DataFrame containing the segmented sample rows under the given index.
    """

    return pd.DataFrame(segmented_rows[index])

def get_labels_segmented_sample(segmented_labels, index):
    """
    Given segmented lables as a numpy array, extracts the segmented sample under the given index.
    
    Args:
        segmented_labels (np.array): The labels of the segmented SVM data in a numpy array
        index (int): The index of the segmented sample labels to get
    
    Returns:
        Pandas DataFrame containing the segmented sample labels under the given index.
    """

    return pd.DataFrame(segmented_labels[index])

###############################################################################
# Format conversion functions
###############################################################################
def convert_36x112_to_36(input):
    """
    Given the 36x112 cols per row svm format, converts it to a 36 cols per row format  

    Args:
        input: The SVM data (without the labels) in numpy array format

    Returns:
        numpy array containing the input in 36 cols per row format
    """
    found_number = -1
    rows,cols = input.shape
    wires_hit = np.zeros((rows,36))
    jend = cols//112
    for i in range(0,rows):
        for j in range(0,jend):
            m = j*112
            for k in range(0,112):
                if input[i][m +k] == 1:
                    if (found_number ==-1):
                        found_number = k
                    else:
                        found_number += k
                        found_number = float(found_number)/2
                        break
                else:
                    if found_number != -1:
                        break
            wires_hit[i][j] = found_number
            found_number = -1
    return wires_hit

def convert_36_to_36x112(input):
    """
    Given the 36 cols per row svm format, converts it to a 36x112 cols per row format  

    Args:
        input: The SVM data (without the labels) in numpy array format
        
    Returns:
        numpy array containing the input in 36x112 cols per row format
    """
    rows,cols = input.shape
    wires_hit = np.zeros((rows,36*112))
    for i in range(0,rows):
        for j in range(0,cols):
            if input[i][j].is_integer():
                wires_hit[i][j*112+int(input[i][j])] = 1
            else:
                wires_hit[i][j*112+int(math.floor(input[i][j]))] = 1
                wires_hit[i][j*112+int(math.ceil(input[i][j]))] = 1

    return wires_hit
