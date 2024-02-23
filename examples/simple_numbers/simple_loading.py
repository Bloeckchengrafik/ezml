from ezml import LoadedModel

if __name__ == '__main__':
    model = LoadedModel("../../run/model.h5")
    while x := input("Enter a number: "):
        print(model.predict([float(x)])[0][0])
