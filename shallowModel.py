import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from os import path
from tqdm import tqdm, trange
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# size = 5000
M = 96  # M for MxM images
train, test, train_labels = [], [], []  # initialize arrays for training images, testing images, and training labels

if path.exists('XtrDrivingImages.npy') and path.exists('YtrDrivingLabels.npy'):
    # if numpy training files exist, load training files
    train_matrix, train_labels = np.load('XtrDrivingImages.npy') / 255.0, np.load('YtrDrivingLabels.npy')

    # remove this
    # idxs = np.random.permutation(train_matrix.shape[0])[0:size]
    # train_matrix, train_labels = train_matrix[idxs,:], train_labels[idxs]

if path.exists('XteDrivingImages.npy') and path.exists('XteDrivingImageNames.npy'):
    # if numpy testing files exist, load testing files
    test_matrix, test_names = np.load('XteDrivingImages.npy'), np.load('XteDrivingImageNames.npy')

    # remove this unless u
    # idxs = np.random.permutation(test_matrix.shape[0])[0:size]
    # test_matrix, test_names = test_matrix[idxs,:] / 255.0, test_names[idxs]

print("Min:{}, Max: {}".format(np.min(train_matrix),
                               np.max(train_matrix)))  # ensure values are normalized (between 0 and 1)

n, m = train_matrix.shape[0], train_matrix.shape[1]
xtrain_flat = train_matrix.reshape(-1, m ** 2)
xtest_flat = test_matrix.reshape(-1, m ** 2)

print('Size of the flattened train_matrix: {}'.format(xtrain_flat.shape))

scaler = preprocessing.MinMaxScaler().fit(xtrain_flat)
x_tr = scaler.transform(xtrain_flat)
scaler = preprocessing.MinMaxScaler().fit(xtest_flat)
x_te = scaler.transform(xtest_flat)

print("scaling done\nstart training")

softReg = LogisticRegression(multi_class='multinomial', solver='sag')
softReg.fit(x_tr, train_labels)

coeffs = softReg.coef_
print('co', coeffs.shape)
print('co', coeffs[0])

predictions = softReg.predict_proba(x_te)

print('pred', predictions.shape)
print('pred', predictions[0])

output = np.vstack((test_names, predictions.transpose())).transpose()

pd.DataFrame(output).to_csv("solution.csv", index=None,
                            header=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
