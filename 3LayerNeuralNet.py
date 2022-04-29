import numpy as np
import pandas

NUM_INPUT = 2304  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons

# Change from 0-9 labels to "one-hot" binary vector labels. For instance,
# if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
def binaryLabels (y):
    newY = np.zeros((10, y.shape[0]))
    for i in range(y.shape[0]):
        newY[y[i]][i] = 1
    return newY

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    # Unpack arguments
    start = 0
    end = NUM_HIDDEN*NUM_INPUT
    W1 = w[0:end]
    start = end
    end = end + NUM_HIDDEN
    b1 = w[start:end]
    start = end
    end = end + NUM_OUTPUT*NUM_HIDDEN
    W2 = w[start:end]
    start = end
    end = end + NUM_OUTPUT
    b2 = w[start:end]
    # Convert from vectors into matrices
    W1 = W1.reshape(NUM_HIDDEN, NUM_INPUT)
    W2 = W2.reshape(NUM_OUTPUT, NUM_HIDDEN)
    return W1,b1,W2,b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return np.hstack((W1.flatten(), b1, W2.flatten(), b2))

def percentCorrect(yhat, y):
    n = y.shape[1]
    yhatMax = np.argmax(yhat, axis=0)
    yMax = np.argmax(y, axis=0)
    correctCount = np.sum(np.equal(yhatMax, yMax))
    return (correctCount / n) * 100

def relu(x):
    copy = np.copy(x)
    copy[copy<=0] = 0
    return copy

def reluPrime(x):
    copy = np.copy(x)
    copy[copy<=0] = 0
    copy[x>0] = 1
    return copy

# Calculates regularization expression
def regularization(w, alpha = 0.):
    W1, b1, W2, b2 = unpack(w)
    w1SquaredSum = np.sum(W1 ** 2)
    w2SquaredSum = np.sum(W2 ** 2)
    return alpha * (w1SquaredSum + w2SquaredSum) / 2

def fwdProp(X,w):
    W1, b1, W2, b2 = unpack(w)
    z1 = W1.dot(X) + b1.reshape(b1.shape[0], 1)
    h1 = relu(z1)
    z2 = W2.dot(h1) + b2.reshape(b2.shape[0], 1)
    yhat = getSoftmaxGuesses(z2)

    return yhat, z1, h1

def backProp(X, Y, w, yhat, z1, h1, alpha = 0.):
    W1, b1, W2, b2 = unpack(w)
    n = X.shape[1]
    gW2 = (yhat - Y).dot(h1.T) / n + (alpha * W2)
    gb2 = np.mean(yhat - Y, axis=1)
    g = (((yhat - Y).T).dot(W2) * reluPrime(z1.T)).T
    gW1 = g.dot(X.T) / n + (alpha * W1)
    gb1 = np.mean(g, axis=1)
    return gW1, gb1, gW2, gb2

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE (X, Y, w, alpha = 0.):
    W1, b1, W2, b2 = unpack(w)
    # print(X, Y, w)

    yhat, z1, h1 = fwdProp(X, w)
    n = Y.shape[1]
    cost = (-1/n) * np.sum(np.sum(Y*np.log(yhat), axis=0)) + regularization(w, alpha)
    acc = percentCorrect(yhat, Y)
    return cost, acc, z1, h1, W1, W2, yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w, yhat, z1, h1, alpha = 0.):
    gW1, gb1, gW2, gb2 = backProp(X, Y, w, yhat, z1, h1, alpha)
    grad = pack(gW1, gb1, gW2, gb2)
    return grad

# returns softmax yhat matrix
def getSoftmaxGuesses(z):
    e = np.exp(z)
    s = np.sum(e, axis=0)
    return e/s

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
# Given training and testing datasets and an initial set of weights/biases b, train the NN.
def train(trainX, trainY, EPOCHS=3, BATCH_SIZE=64, EPSILON=0.0025, HIDDEN=NUM_HIDDEN):
    m, n = trainX.shape
    NUM_HIDDEN = HIDDEN
    # Initialize weights randomly
    W1 = 2 * (np.random.random(size=(HIDDEN, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    b1 = 0.01 * np.ones(HIDDEN)
    W2 = 2 * (np.random.random(size=(NUM_OUTPUT, HIDDEN)) / HIDDEN ** 0.5) - 1. / HIDDEN ** 0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    for e in range(EPOCHS):
        rand_i = np.random.permutation(n)  # shuffle indices
        Xtrain = trainX[:, rand_i]  # new random order of columns in X
        Ytrain = trainY[:, rand_i]  # matching random order of associated ground truth labels

        for i in range((n // BATCH_SIZE) - 1):  # loop through (n / ntilde) - 1 times
            blow, bupp = BATCH_SIZE * i, BATCH_SIZE * (i + 1)  # upper and lower batch bounds
            X_batch, Y_batch = Xtrain[:, blow:bupp], Ytrain[:, blow:bupp]  # associated X and Y batches

            yhat, z1, h1 = fwdProp(X_batch, w)
            grad = gradCE(X_batch, Y_batch, w, yhat, z1, h1)  # calculate gradient terms
            gW1, gb1, gW2, gb2 = unpack(grad)  # unpack gradient terms
            W1, b1, W2, b2 = unpack(w)

            # update weights and biases
            W1 = W1 - EPSILON * gW1
            b1 = b1 - EPSILON * gb1
            W2 = W2 - EPSILON * gW2
            b2 = b2 - EPSILON * gb2

            # repack weights and biases
            w = pack(W1, b1, W2, b2)  # recalculate weights



    return w

def findBestHyperparameters (trainX, trainY, testX, testY):
    bestLoss = 100
    bestHl = 0
    bestLr = 0
    bestMb = 0
    bestE =  0
    bestA =  0
    bestW = None
    for i in range(10):
        hl = np.random.choice(hiddenLayers)
        lr = np.random.choice(learningRates)
        mb = np.random.choice(minibatches)
        e = np.random.choice(epochs)
        a = np.random.choice(alphas)
        NUM_HIDDEN = hl
        print(hl, lr, mb, e, a)
        trainedW = train(trainX, trainY, testX, testY, e, mb, lr, hl)
        fce = fCE(testX, testY, trainedW, a)
        if fce[0] < bestLoss:
            bestLoss = fce[0]
            bestHl = hl
            bestLr = lr
            bestMb = mb
            bestE = e
            bestA = a
            bestW = trainedW
        testYhat = fwdProp(testX, trainedW)[0]
        print("test percent correct: ", percentCorrect(testYhat, testY))
        print(percentCorrect(testYhat, testY))

    print("Best Parameters: ")
    print("best # Hidden Layers:", bestHl)
    print("best learning rate:", bestLr)
    print("best batch size:", bestMb)
    print("best # epochs:", bestE )
    print("best alpha:", bestA )

    testYhat = fwdProp(testX, bestW)[0]
    print("test percent correct: ", percentCorrect(testYhat, testY))

if __name__ == "__main__":
    # Load data
    X = np.load("XtrDrivingImages.npy".format("train")).T / 255.
    Y = binaryLabels(np.load("YtrDrivinglabels.npy".format("train")))
    X_test = np.load("XteDrivingImages.npy".format("test")).T / 255.
    test_names = np.load("XteDrivingImageNames.npy")

    X = X.reshape(2304,22424)
    X_test = X_test.reshape(2304,79726)
    # Train the network using SGD.
    w = train(X, Y, 100, 256, 0.01, 40)

    fce = fCE(X,Y, w)

    print("loss: ", fce[0], "    acc: ", fce[1])

    #visualization
    # img = X[:,:,0]
    # plt.imshow(img)
    # plt.show()

    # Save to csv
    yhat, z1, h1 = fwdProp(X_test, w)

    results = {'img': test_names}

    for i in range(10):
        results['c' + str(i)] = yhat[i]

    output = pandas.DataFrame(results)
    output.to_csv('submission.csv', index=False)

