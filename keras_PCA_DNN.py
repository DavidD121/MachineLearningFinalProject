import numpy as np
import pandas as pd
import time
from dataPCA import load_data
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop


# calculate set of PCs of given flattened xtrain and xtest that achieve explained variance = var(default 0.9)
def getPCA(xtrain_flat, xtest_flat, var=0.9):
    # find n_components that achieve 90% explained variance
    pca = PCA(var)
    pca.fit(xtrain_flat)
    print("{} PCs explain {}% of variance".format(pca.n_components_, var * 100))

    # apply transform on train & test set to generate new dataset from the calculated PCs
    train_img_pca = pca.transform(xtrain_flat)
    test_img_pca = pca.transform(xtest_flat)

    return train_img_pca, test_img_pca


if __name__ == "__main__":
    begin = time.time()  # program start time
    batch_size = 128
    num_classes = 10
    epochs = 20

    # Load data and normalize images
    xtrain = load_data('XtrDrivingImages.npy') / 255.0
    ytrain = load_data('YtrDrivingLabels.npy')
    xtest = load_data('XteDrivingImages.npy') / 255.0
    test_image_names = load_data('XteDrivingImageNames.npy').reshape(79726, 1)

    print("Min:{}, Max: {}".format(np.min(xtrain), np.max(xtrain)))  # ensure values are normalized (between 0 and 1)

    # ---------------------------------------------------------------------------------------------------------------
    # Leveraging PCA to create a usable data that achieves a desired variance, to be passed to a learning model.
    # ---------------------------------------------------------------------------------------------------------------

    n, m = xtrain.shape[0], xtrain.shape[1]
    xtrain_flat = xtrain.reshape(-1, m ** 2)
    xtest_flat = xtest.reshape(-1, m ** 2)

    # get calculated PC datasets of train and test sets
    train_img_pca, test_img_pca = getPCA(xtrain_flat, xtest_flat, .95)

    # train labels to one-hot encoding
    ytrain = np_utils.to_categorical(ytrain)

    # Build 5 layer Sequential model
    model = Sequential()
    model.add(Dense(2592, activation='relu', input_shape=(290,)))
    model.add(Dense(2592, activation='relu'))
    model.add(Dense(1296, activation='relu'))
    model.add(Dense(648, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # Train model and store History object of trained model
    history = model.fit(train_img_pca, ytrain, batch_size=batch_size, epochs=epochs,
                        verbose=1, shuffle=True, use_multiprocessing=True)
    # print("History:\n{}".format(history))

    # Use trained model to predict test data
    predictions = model.predict(test_img_pca, batch_size=batch_size, verbose=1, use_multiprocessing=True)
    print("First 10 Predictions:\n{}".format(predictions[:10]))

    # Save predictions as .npy file
    np.save('ClassPredictions.npy', predictions)

    # Add labels with predictions
    pred = np.append(test_image_names, predictions, 1)

    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(pred)

    # Convert DataFrame to CSV
    pred_df.to_csv('DNN_PCA_Predictions.csv', index=False,
                   columns=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])

    end = time.time()  # program end time
    print("Execution Time: {} seconds".format(round(end - begin, 2)))
