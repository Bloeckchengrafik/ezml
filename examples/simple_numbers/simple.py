import numpy as np

import ezml
from ezml import layers


def data():
    for i in range(10000):
        yield i, i * 2


def test_data():
    for i in range(100):
        yield i, i * 2


class MySimpleModel(ezml.Model):
    def build(self):
        with self.sequential() as s:
            s += layers.Dense(2, input_shape=(1,))
            s += layers.Dense(1)


def main():
    model = MySimpleModel()
    model.summary()

    model.compile("adam", "mse")
    model.fit(ezml.DataDeclaration(
        data=data,
        test_data=test_data,
        steps_per_epoch=10000,
        validation_steps=100
    ), epochs=10)

    model.save("model.h5")

    while True:
        x = input("Enter a number: ")
        print(model.predict(np.ndarray([float(x)])))


if __name__ == "__main__":
    main()
