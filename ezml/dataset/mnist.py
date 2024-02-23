from ezml.data import DataDeclaration
from ezml.dataset.dataset import Dataset
from tensorflow.keras.datasets import mnist


class MNISTDataset(Dataset):
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        # Download the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.loaded = True

    def train(self):
        for x, y in zip(self.x_train, self.y_train):
            yield x, y

    def test(self):
        for x, y in zip(self.x_test, self.y_test):
            yield x, y

    def declare(self) -> DataDeclaration:
        return DataDeclaration(
            data=self.train,
            test_data=self.test,
            steps_per_epoch=len(self.x_train),
            validation_steps=len(self.x_test)
        )

    def get_sample(self):
        self.load()
        return next(self.test())

