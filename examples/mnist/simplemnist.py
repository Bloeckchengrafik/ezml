from ezml import Model, dataset, layers, model


def mnist(m: Model):
    with m.sequential() as s:
        s += layers.Flatten(input_shape=(28, 28))
        s += layers.Dense(128, activation="relu")
        s += layers.Dense(10, activation="softmax")


if __name__ == '__main__':
    model = model(mnist).compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(dataset.mnist(), epochs=1)
    model.save("mnist.h5")

    x, y = dataset.mnist().get_sample()
    data = model.predict(x)
    print(y, data)
