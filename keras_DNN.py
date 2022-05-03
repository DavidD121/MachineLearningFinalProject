import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from dataPCA import load_data
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

if __name__ == "__main__":
    begin = time.time()  # program start time
    batch_size = 64
    num_classes = 10
    epochs = 20
    epsilon = 1e-7
    learn_rate = 1e-3
    # loss_func = RMSprop()
    loss_func = SGD(learning_rate=learn_rate)
    # loss_func = Adam(epsilon=epsilon, learning_rate=learn_rate)

    # Load data and normalize images
    xtrain = load_data('XtrDrivingImages.npy') / 255.0
    ytrain = load_data('YtrDrivingLabels.npy')
    xtest = load_data('XteDrivingImages.npy') / 255.0
    test_image_names = load_data('XteDrivingImageNames.npy').reshape(79726, 1)

    print("Min:{}, Max: {}".format(np.min(xtrain), np.max(xtrain)))  # ensure values are normalized (between 0 and 1)

    # Get shape of train matrix and flatten train and test
    n, m = xtrain.shape[0], xtrain.shape[1]
    xtrain_flat = xtrain.reshape(-1, m ** 2)
    xtest_flat = xtest.reshape(-1, m ** 2)

    # train labels to one-hot encoding
    ytrain = np_utils.to_categorical(ytrain)

    # Build 5 layer Sequential model
    model = Sequential()
    model.add(Dense(2592, activation='relu', input_shape=(5184,)))
    model.add(Dense(2592, activation='relu'))
    model.add(Dense(1296, activation='relu'))
    model.add(Dense(648, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=loss_func,
                  metrics=['accuracy'])

    # Train model and store History object of trained model
    history = model.fit(xtrain_flat, ytrain, batch_size=batch_size, epochs=epochs,
                        verbose=1, shuffle=True, use_multiprocessing=True)
    # print("History:\n{}".format(history))

    # Use trained model to predict test data
    predictions = model.predict(xtest_flat, batch_size=batch_size, verbose=1, use_multiprocessing=True)
    # print("First 10 Predictions:\n{}".format(predictions[:10]))

    # Save predictions as .npy file
    np.save('ClassPredictions.npy', predictions)

    # Add labels with predictions
    pred = np.append(test_image_names, predictions, 1)

    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(pred)

    cnames = {0: 'img', 1: 'c0', 2: 'c1', 3: 'c2', 4: 'c3', 5: 'c4',
              6: 'c5', 7: 'c6', 8: 'c7', 9: 'c8', 10: 'c9'}  # column names
    pred_df = pred_df.rename(cnames, axis='columns')  # rename columns for submission

    # Convert DataFrame to CSV
    pred_df.to_csv('DNN_Predictions_BS{}_E{}_LR{}.csv'.format(batch_size, epochs, learn_rate), index=False)

    end = time.time()  # program end time
    print("Execution Time: {} seconds".format(round(end - begin, 2)))

    # ----------------------------------------
    # PLOT ACCURACY & LOSS OF TRAINING MODEL
    # ----------------------------------------
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
