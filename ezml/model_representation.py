import abc


class ModelRepresentation(abc.ABC):
    @abc.abstractmethod
    def into(self):
        pass
