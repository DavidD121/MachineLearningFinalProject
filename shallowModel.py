import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from os import path
from tqdm import tqdm, trange
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

M = 96  # M for MxM images
train, test, test_labels, train_labels = [], [], [], []  # initialize arrays for training images, testing images, and training labels

if path.exists('XtrDrivingImages.npy') and path.exists('YtrDrivingLabels.npy'):
    # if numpy training files exist, load training files
    train_matrix, train_labels = np.load('XtrDrivingImages.npy') / 255.0, np.load('YtrDrivingLabels.npy')
    #train_matrix, train_labels = train_matrix[0:3000], train_labels[0:3000] #TODO: remove this

if path.exists('XteDrivingImages.npy') and path.exists('XteDrivingImageNames.npy'):
    # if numpy testing files exist, load testing files
    test_matrix, test_labels = np.load('XteDrivingImages.npy'), np.load('XteDrivingImageNames.npy')
    #test_matrix, test_labels = test_matrix[0:3000] / 255.0, test_labels[0:3000] #TODO: remove this


print("Min:{}, Max: {}".format(np.min(train_matrix), np.max(train_matrix)))  # ensure values are normalized (between 0 and 1)

n, m = train_matrix.shape[0], train_matrix.shape[1]
xtrain_flat = train_matrix.reshape(-1, m ** 2)
xtest_flat = test_matrix.reshape(-1, m ** 2)

print('Size of the flattened train_matrix: {}'.format(xtrain_flat.shape))

scaler = preprocessing.StandardScaler().fit(xtrain_flat)
x_tr = scaler.transform(xtrain_flat)
scaler = preprocessing.StandardScaler().fit(xtest_flat)
x_te = scaler.transform(xtest_flat)

print(x_tr.shape)
print(train_labels.shape)

softReg = LogisticRegression(multi_class = 'multinomial', solver = 'sag')
softReg.fit(x_tr, train_labels)

score = softReg.score(x_te, test_labels)
print(score)
