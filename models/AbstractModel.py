from abc import ABC, abstractmethod

class AbstractModel(ABC):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = None

    @abstractmethod
    def load_model(self, path):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path):
        raise NotImplementedError

    @abstractmethod
    def build_new_model(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess_input(self, input_dict):
        raise NotImplementedError

    @abstractmethod
    def compute_accuracy_metrics(self, input_dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def train(self, input_dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def test(self, input_dict) -> dict:
        raise NotImplementedError

    # TODO finalize prototype
    @abstractmethod
    def predict(self):
        raise NotImplementedError