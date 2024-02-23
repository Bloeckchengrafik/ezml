from abc import ABC, abstractmethod

from ezml.data import DataDeclaration


class Dataset(ABC):
    """
    Load a default dataset
    """

    @abstractmethod
    def load(self):
        """
        Load the dataset
        """
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def declare(self) -> DataDeclaration:
        pass
