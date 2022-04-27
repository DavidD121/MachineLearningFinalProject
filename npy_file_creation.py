import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm, trange

train, test, labels = [], [], []  # initialize arrays for training images, testing images, and training labels

# loop through all training images and store images and labels in associated arrays (train, labels)
for _ in range(10):
    for filename in tqdm(glob.glob('imgs/train/c{}/*.jpg'.format(_))):  # assuming gif
        im = np.asarray(Image.open(filename).convert('L').resize((256, 256)))
        train.append(im)
        labels.append(_)

# loop through all testing images and store in associated array (test)
for filename in tqdm(glob.glob('imgs/test/*.jpg')):  # assuming gif
    im = np.asarray(Image.open(filename).convert('L').resize((256, 256)))
    test.append(im)

train_matrix, label_matrix, test_matrix = np.array(train), np.array(labels), np.array(test)
np.save("XtrDrivingImages", train_matrix)  # output training images
np.save("YtrDrivingLabels", label_matrix)  # output training labels
np.save("XteDrivingImages", test_matrix)  # output testing images

print("Shape of Train Matrix: ", train_matrix.shape,
      "Shape of Label Matrix: ", label_matrix.shape,
      "Shape of Test Matrix: ", test_matrix.shape)

img = train[0]
fig, ax = plt.subplots(1)
ax.imshow(img, cmap='gray')
plt.show()
