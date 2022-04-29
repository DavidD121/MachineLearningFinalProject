import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from os import path
from tqdm import tqdm, trange

M = 96  # M for MxM images
train, test, test_image_names, labels = [], [], [], []  # initialize arrays for training images, testing images, and training labels

if path.exists('XtrDrivingImages.npy') and path.exists('YtrDrivingLabels.npy'):
    # if numpy training files exist, load training files
    train_matrix, label_matrix = np.load('XtrDrivingImages.npy'), np.load('YtrDrivingLabels.npy')

else:
    # loop through all training images and store images and labels in associated arrays (train, labels)
    for _ in range(10):
        for filename in tqdm(glob.glob('imgs/train/c{}/*.jpg'.format(_))):
            im = np.asarray(Image.open(filename).convert('L').resize((M, M)))
            train.append(im)
            labels.append(_)

    # Convert image and labels lists to numpy arrays (matrix and vector)
    train_matrix, label_matrix = np.array(train), np.array(labels)

    # Training Matricies
    np.save("XtrDrivingImages", train_matrix)  # output training images
    np.save("YtrDrivingLabels", label_matrix)  # output training labels

if path.exists('XteDrivingImages.npy') and path.exists('XteDrivingImageNames.npy'):
    # if numpy testing files exist, load testing files
    test_matrix, test_name_matrix = np.load('XteDrivingImages.npy'), np.load('XteDrivingImageNames.npy')

else:
    # loop through all testing images and store in associated array (test)
    for filename in tqdm(glob.glob('imgs/test/*.jpg')):
        im = np.asarray(Image.open(filename).convert('L').resize((M, M)))
        test.append(im)  # append formatted image to
        im_name = filename.replace('imgs/test/', '').strip()  # get image name
        test_image_names.append(im_name)  # append to image names list

    # Convert image and name lists to numpy arrays (matrix and vector)
    test_matrix, test_name_matrix = np.array(test), np.array(test_image_names)

    # Testing Matricies
    np.save("XteDrivingImages", test_matrix)  # output testing images
    np.save("XteDrivingImageNames", test_image_names)  # output testing image names

# See shape of all matricies / vectors
print("Shape of Train Matrix: ", train_matrix.shape,
      "Shape of Label Matrix: ", label_matrix.shape,
      "Shape of Test Matrix: ", test_matrix.shape,
      "Shape of Test Name Matrix: ", test_name_matrix.shape)

# view n random images in training matrix
n = 5
idxs = np.random.permutation(train_matrix.shape[0])[0:n]
for i in idxs:
    img = train_matrix[i]
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
plt.show()
