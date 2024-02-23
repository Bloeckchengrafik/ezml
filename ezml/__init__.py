from abc import ABC, abstractmethod
from typing import Union, Callable

import numpy as np

from ezml.dataset.dataset import Dataset
from ezml.keras import KerasModel
from ezml.sequential import Sequential
from ezml.data import DataDeclaration

from tensorflow.keras import Model as KModel


class Model(ABC):
    def __init__(self, path=None):
        self.compile_args = []
        self.compile_kwargs = {}
        self.path = path
        self.model = None
        self.compiled = None

        if path is not None:
            self.load()
        else:
            self.build()

    @abstractmethod
    def build(self):
        pass

    def load(self):
        # load model from path
        self.model = KerasModel(self.path)

    def sequential(self) -> Sequential:
        self.model = Sequential()

        return self.model

    def into(self) -> KModel:
        return self.model.into()

    def save(self, path):
        self.model.into().save(path)

    def summary(self):
        self.model.into().summary()

    def compile(self, *args, **kwargs) -> 'Model':
        self.compile_args = args
        self.compile_kwargs = kwargs

        model = self.into()
        model.compile(*self.compile_args, **self.compile_kwargs)

        return self

    def fit(self, data: Union[DataDeclaration, Dataset], **kwargs) -> 'Model':
        if self.model is None:
            raise Exception("Model not built.")

        if len(self.compile_kwargs.items()) == 0 and len(self.compile_args) == 0:
            raise Exception("Model not compiled.")

        if isinstance(data, Dataset):
            data.load()
            data = data.declare()

        model = self.into()
        model.compile(*self.compile_args, **self.compile_kwargs)

        def _wrapper_data(fn):
            while True:
                for x, y in fn():
                    if isinstance(x, list):
                        yield x, y
                    yield np.array([x]), np.array([y])

        model.fit(
            _wrapper_data(data.data),
            validation_data=_wrapper_data(data.test_data),
            steps_per_epoch=data.steps_per_epoch,
            validation_steps=data.validation_steps,
            **kwargs
        )

        return self

    def predict(self, param):
        if self.model is None:
            raise Exception("Model not built.")

        return self.into()(param)


class LoadedModel(Model):
    def build(self):
        raise NotImplementedError("LoadedModel cannot be built.")


class FunctionalModel(Model):
    def __init__(self, function: Callable[[Model], None]):
        self.fn = function
        super().__init__()

    def build(self):
        self.fn(self)


def model(dt: Union[Callable[[Model], None], str]) -> Model:
    if isinstance(dt, str):
        return LoadedModel(dt)
    return FunctionalModel(dt)
