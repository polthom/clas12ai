#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse
import os.path
from sklearn.metrics import confusion_matrix
from termcolor import colored

from utils.svm_utils import *
from models.ExtraTreesModel import ExtraTreesModel
from models.CnnModel import CnnModel
from models.MlpModel import MlpModel

## Script entry point
def main():

    args = parse_arguments()
    print(args)

    subroutine = get_subroutine(args)
    subroutine(args)

def parse_arguments():
    """
    Parse CLI arguments and return an object containing the values.
    """

    # Create main program parser
    parser = argparse.ArgumentParser(description="CRTC-JLab Machine Learning CLI")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "subprogram"

    # Create training subprogram parser
    parser_train = subparsers.add_parser("train", help="Train a model, perform testing, and serialize it.")
    parser_train._action_groups.pop()
    parser_train_required_args = parser_train.add_argument_group("Required arguments")
    parser_train_optional_args = parser_train.add_argument_group("Optional arguments")
    parser_train_required_args.add_argument("--training-dir", "-t", required=True, help="Path to the directory containing the training data.", dest="training_dir_path")
    parser_train_required_args.add_argument("--testing-dir", "-e", required=True, help="Path to the directory containing the testing data.", dest="testing_dir_path")
    parser_train_required_args.add_argument("--num-features", "-f", choices=["6", "36", "4032"], required=True, help="Path to the directory containing the testing data.", dest="num_features")
    parser_train_required_args.add_argument("--out-model", "-m", required=True, help="Name of the file in which to save the model.", dest="output_model_path")
    parser_train_required_args.add_argument("--model-type", choices=["cnn", "mlp", "et"], required=True, help="The type of the model to train.", dest="model_type")
    parser_train_optional_args.add_argument("--epochs", required=False, type=int, default="5", help="How many training epochs to go through.", dest="training_epochs")
    parser_train_optional_args.add_argument("--batchSize", required=False, type=int, default="16", help="Size of the training batch.", dest="training_batch_size")
    parser_train_optional_args.add_argument("--testing-batchSize", required=False, type=int, default="16", help="Size of the evaluation batch.", dest="evaluation_batch_size")

    # Create evaluation subprogram parser
    parser_test = subparsers.add_parser("test", help="Load a model for testing.")
    parser_test._action_groups.pop()
    parser_test_required_args = parser_test.add_argument_group("Required arguments")
    parser_test_optional_args = parser_test.add_argument_group("Optional arguments")
    parser_test_required_args.add_argument("--testing-dir", "-e", required=True, help="Path to the directory containing the testing data.", dest="testing_dir_path")
    parser_test_required_args.add_argument("--num-features", "-f", choices=["6", "36", "4032"], required=True, help="Path to the directory containing the testing data.", dest="num_features")
    parser_test_required_args.add_argument("--model", "-m", required=True, help="The name of the file from which to load the model.", dest="model_path")
    parser_test_required_args.add_argument("--model-type", choices=["cnn", "mlp", "et"], required=True, help="The type of the model to load.", dest="model_type")
    parser_test_optional_args.add_argument("--batchSize", required=False, type=int, default="16", help="Size of the evaluation batch.", dest="evaluation_batch_size")


    # Create prediction subprogram parser
    parser_predict = subparsers.add_parser("predict", help="Load a model and use it for predictions.")
    parser_predict._action_groups.pop()
    parser_predict_required_args = parser_predict.add_argument_group("Required arguments")
    parser_predict_optional_args = parser_predict.add_argument_group("Optional arguments")
    parser_predict_required_args.add_argument("--prediction-dir", "-p", required=True, help="Path to the directory containing the prediction data.", dest="prediction_dir_path")
    parser_predict_required_args.add_argument("--prediction-output-file", "-o", required=True, help="File in which to save the predictions.", dest="prediction_output_file")
    parser_predict_required_args.add_argument("--model", "-m", required=True, help="The name of the file from which to load the model.", dest="model_path")
    parser_predict_required_args.add_argument("--model-type", choices=["cnn", "mlp", "et"], required=True, help="The type of the model to load.", dest="model_type")
    parser_predict_optional_args.add_argument("--batchSize", required=False, type=int, default="32", help="Size of the prediction batch.", dest="prediction_batch_size")

    return parser.parse_args()

def enumerate_data_files_in_dir(dir_path):
    return glob.glob(f'{dir_path}/*.txt')

def read_input_data(input_type, args) -> dict:
    """
    Reads all of the SVM files in the training and testing directories,
    concatenates them, and creates a dictionary that references them.

    Args:
        input_type: Type of reading and processing of the input for training, evaluation, or prediction.
        args: The object that contains all the parsed CLI arguments. 

    Returns:
        A dictionary containing the read and processed input data. This includes the features and targets when appropriate.
    """

    if input_type == "train":
        # Read training and testing data
        training_dir_filenames = enumerate_data_files_in_dir(args.training_dir_path)
        testing_dir_filenames = enumerate_data_files_in_dir(args.testing_dir_path)

        X_train, y_train = read_concat_svm_files(training_dir_filenames, int(args.num_features))
        X_test, y_test = read_concat_svm_files(testing_dir_filenames, int(args.num_features))

        X_test_segmented, y_test_segmented = segment_svm_data(X_test, y_test)
        total_test_samples = len(y_test_segmented)
        
        return {
            "training": {"data": X_train, "labels": y_train, "epochs": args.training_epochs,"batch_size": args.training_batch_size}, 
            "testing": {"data": X_test, "labels": y_test,"batch_size": args.evaluation_batch_size},
            "testing_segmented": {"data": X_test_segmented, "labels": y_test_segmented},
            "total_test_samples": total_test_samples
        }

    elif input_type == "test":
        # Read testing data
        testing_dir_filenames = enumerate_data_files_in_dir(args.testing_dir_path)

        X_test, y_test = read_concat_svm_files(testing_dir_filenames, int(args.num_features))

        X_test_segmented, y_test_segmented = segment_svm_data(X_test, y_test)

        total_test_samples = len(y_test_segmented)

        return {
            "testing": {"data": X_test, "labels": y_test, "batch_size": args.evaluation_batch_size},
            "testing_segmented": {"data": X_test_segmented, "labels": y_test_segmented},
            "total_test_samples": total_test_samples
        }
    
    #elif input_type == "predict":
    #    return {"prediction": {"data": prediction_data}}

    else:
        print(colored("Error: Wrong input type.", "red"))
        quit()

def print_training_report(training_dict):
    training_time = training_dict["training_time"]
    training_accuracy = training_dict["accuracy_training"]

    print("\nTraining Report")
    print("================================")
    print(colored(f'Training accuracy: {training_accuracy}', "blue"))
    print(colored(f'Training time: {training_time}s', "blue"))

def print_testing_report(testing_dict):
    testing_accuracy = testing_dict["accuracy_testing"]
    testing_prediction_time = testing_dict["testing_prediction_time"]
    conf_matrix = testing_dict["confusion_matrix"]

    accuracy_A1 = testing_dict["accuracy_A1"]
    accuracy_Ac = testing_dict["accuracy_Ac"]
    accuracy_Ah = testing_dict["accuracy_Ah"]
    accuracy_Af = testing_dict["accuracy_Af"]

    print("\nTesting Report")
    print("================================")
    print(colored(f'Testing accuracy: {testing_accuracy}', "blue"))
    print(colored(f'Testing prediction time: {testing_prediction_time}s', "blue"))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print()
    print(colored(f'Accuracy A1: {accuracy_A1}', "yellow"))
    print(colored(f'Accuracy Ac: {accuracy_Ac}', "yellow"))
    print(colored(f'Accuracy Ah: {accuracy_Ah}', "yellow"))
    print(colored(f'Accuracy Af: {accuracy_Af}', "yellow"))

def train_model(args):
    """
    Trains the model with the input data specified in the CLI arguments followed by an evaluation of the model.

    Args:
        args: The object that contains all the parsed CLI arguments. 
    """

    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("train", args)

    model = None

    if args.model_type == "et":
        model = ExtraTreesModel()
    elif (args.model_type == "mlp"):
        model = MlpModel()
    elif (args.model_type == "cnn"):
        model = CnnModel(in_dict = input_dict)

    model.build_new_model()
    training_dict = model.train(input_dict)

    testing_dict = model.test(input_dict)
    
    print_training_report(training_dict)
    print_testing_report(testing_dict)

    model.save_model(args.output_model_path)

def test_model(args):
    """
    Evaluates the model with the input data specified in the CLI arguments.

    Args:
        args: The object that contains all the parsed CLI arguments. 
    """

    print(colored("\nReading input data...", "green"))
    input_dict = read_input_data("test", args)

    model = None

    if (args.model_type == "et"):
        model = ExtraTreesModel()
    elif (args.model_type == "mlp"):
        model = MlpModel()
    elif (args.model_type == "cnn"):
        model = CnnModel(in_dict = input_dict)
    
    model.load_model(args.model_path)

    testing_dict = model.test(input_dict)

    print_testing_report(testing_dict)

# TODO implement
def predict(args):
    if (args.model_type == "et"):
        model = ExtraTreesModel()
    elif (args.model_type == "mlp"):
        model = MlpModel()
    elif (args.model_type == "cnn"):
        model = CnnModel()

    model.load_model(args.model_path)

    # TODO continue implementation
    raise NotImplementedError

def get_subroutine(args):
    """
    Processes the CLI subprogram utilized and return the matching function for training, evaluation, or prediction.

    Args:
        args: The object that contains all the parsed CLI arguments. 

    Returns:
        The functions corresponding the the CLI subprogram utilized.
    """

    if args.subprogram == "train":
        print(colored("Executing training subprogram.", "green"))
        return train_model
    elif args.subprogram == "test":
        print(colored("Executing testing subprogram.", "green"))
        return test_model
    elif args.subprogram == "predict":
        print(colored("Executing prediction subprogram.", "green"))
        return predict
    else:
        print(colored("Fatal Error: Wrong subprogram specified.", "red"))
        quit()

if __name__ == "__main__": main()
