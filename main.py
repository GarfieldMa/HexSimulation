from experiment import *


if __name__ == '__main__':

    # params = [, "w1ww0.json", "w0ww1.json"]
    params = ["w0ww0.json"]

    for param in params:
        e = Experiment(param)
        e.run()