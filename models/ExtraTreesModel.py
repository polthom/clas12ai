from sklearn.ensemble import ExtraTreesClassifier
from models.AbstractScikitLearnClassifier import AbstractScikitLearnClassifier
from termcolor import colored

class ExtraTreesModel(AbstractScikitLearnClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def build_new_model(self):
        self.model = ExtraTreesClassifier(n_estimators=300, criterion='entropy', max_features=None,
                                   n_jobs=-1, verbose=1, random_state=3333)

    def train(self, input_dict) -> dict:
        print(colored("Training ExtraTrees model...", "green"))
        return super().train(input_dict)

    def test(self, input_dict) -> dict:
        print(colored("Testing ExtraTrees model...", "green"))
        return super().test(input_dict)