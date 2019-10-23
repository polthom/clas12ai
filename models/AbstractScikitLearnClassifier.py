from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from termcolor import colored
import pickle

from models.AbstractModel import AbstractModel
from utils.accuracy_utils import *

class AbstractScikitLearnClassifier(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def load_model(self, path):
        print(colored(f'\nLoading scikit-learn model from {path}\n', "green"))
        self.model = pickle.load(open(f'{path}', "rb"))

    def save_model(self, path):
        print(colored(f'\nSaving scikit-learn model in {path}.p', "green"))
        pickle.dump(self.model, open(f'{path}.p', "wb"))

    def preprocess_input(self, input_dict):
        None

    def compute_accuracy_metrics(self, input_dict) -> dict:
        conf_matrix = input_dict["confusion_matrix"]
        total_test_samples = input_dict["total_test_samples"]
        y_pred = input_dict["y_pred"]
        y_pred_proba = input_dict["y_pred_proba"]
        y_test_segmented = input_dict["y_test_segmented"]

        accuracy_A1 = get_accuracy_A1(conf_matrix, total_test_samples)
        accuracy_Ac = get_accuracy_Ac(conf_matrix, y_pred, y_test_segmented)
        accuracy_Ah = get_accuracy_Ah(y_pred_proba, y_test_segmented)
        accuracy_Af = get_accuracy_Af(conf_matrix, total_test_samples)

        return {
            "accuracy_A1": accuracy_A1,
            "accuracy_Ac": accuracy_Ac,
            "accuracy_Ah": accuracy_Ah,
            "accuracy_Af": accuracy_Af
        }

    def train(self, input_dict) -> dict:
        X_train = input_dict["training"]["data"]
        y_train = input_dict["training"]["labels"]

        start = timer()
        self.model.fit(X_train, y_train)
        end = timer()
        training_time = end - start

        accuracy_training = self.model.score(X_train, y_train)

        return {
            "training_time": training_time,
            "accuracy_training": accuracy_training
        }

    def test(self, input_dict) -> dict:

        X_test = input_dict["testing"]["data"]
        y_test = input_dict["testing"]["labels"]

        accuracy_testing = self.model.score(X_test, y_test)

        start = timer()
        y_pred = self.model.predict(X_test)
        end = timer()
        testing_prediction_time = end - start

        y_pred_proba = self.model.predict_proba(X_test)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Compute accuracy metrics
        input_dict_accuracy_metrics = {
            "confusion_matrix": conf_matrix, 
            "y_pred": y_pred, 
            "y_pred_proba": y_pred_proba,
            "total_test_samples": input_dict["total_test_samples"], 
            "y_test_segmented": input_dict["testing_segmented"]["labels"]
        }

        accuracy_metrics_dict = self.compute_accuracy_metrics(input_dict_accuracy_metrics)

        output_dict = {
            "accuracy_testing": accuracy_testing, 
            "testing_prediction_time": testing_prediction_time, 
            "confusion_matrix": conf_matrix
        }
        output_dict.update(accuracy_metrics_dict)

        return output_dict

    # TODO implement
    def predict(self):
        None