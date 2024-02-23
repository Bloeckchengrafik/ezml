from ezml.model_representation import ModelRepresentation
from tensorflow.keras.models import load_model


class KerasModel(ModelRepresentation):
    """
    Load a keras model from a file.
    """

    def __init__(self, file_path):
        self.model = load_model(file_path)

    def into(self):
        return self.model
