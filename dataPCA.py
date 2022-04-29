import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_data(data):
    return np.load(data)


if __name__ == "__main__":

    # Load data
    Xtrain = load_data('XtrDrivingImages.npy') / 255
    Ytrain = load_data('YtrDrivingLabels.npy')

    print("Min:{}, Max: {}".format(np.min(Xtrain), np.max(Xtrain)))




