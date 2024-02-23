from ezml.model_representation import ModelRepresentation
from tensorflow.keras.models import Sequential as KSequential


class Sequential(ModelRepresentation):
    def __init__(self):
        self.layers = []

    def into(self):
        return KSequential(self.layers)

    def add(self, layer):
        self.layers.append(layer)

    def __iadd__(self, other):
        self.add(other)

        return self

    def __enter__(self) -> 'Sequential':
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
