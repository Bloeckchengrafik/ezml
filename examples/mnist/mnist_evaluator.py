from ezml.prelude import *

if __name__ == '__main__':
    model = model("mnist.h5").compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    x, y = dataset.mnist().get_sample(1)
    data = model.predict([x])
    print(y, data)
