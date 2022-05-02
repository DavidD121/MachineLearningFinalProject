import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tqdm import trange


# load data from .npy file
def load_data(data):
    return np.load(data)


if __name__ == "__main__":

    begin = time.time()  # program start time

    # Load data and normalize images
    xtrain = load_data('XtrDrivingImages.npy') / 255.0
    ytrain = load_data('YtrDrivingLabels.npy')
    xtest = load_data('XteDrivingImages.npy') / 255.0
    # test_image_names = load_data('XteDrivingImageNames.npy')

    print("Min:{}, Max: {}".format(np.min(xtrain), np.max(xtrain)))  # ensure values are normalized (between 0 and 1)

    n, m = xtrain.shape[0], xtrain.shape[1]
    xtrain_flat = xtrain.reshape(-1, m ** 2)
    xtest_flat = xtest.reshape(-1, m ** 2)
    print('Size of the flattened xtrain: {}'.format(xtrain_flat.shape))

    feat_cols = ['pixel' + str(i) for i in range(xtrain_flat.shape[1])]
    df_train = pd.DataFrame(xtrain_flat, columns=feat_cols)
    df_train['label'] = ytrain
    print('Size of the dataframe: {}'.format(df_train.shape))  # show size=

    # ---------------------------------------------------------------------------------------------------------------
    # Perform PCA(n_components=2) on training set and visualize
    # ---------------------------------------------------------------------------------------------------------------

    # initialize PCA model
    pca_train = PCA(n_components=2)
    # fit PCA model with DataFrame data from training matrix
    train_pc = pca_train.fit_transform(df_train.iloc[:, :-1])

    n_pc = []  # list of column names for principal component DataFrame
    for n in range(2):
        n_pc.append('PC {}'.format(n + 1))

    # create DataFrame for principal components
    train_pc_df = pd.DataFrame(data=train_pc, columns=n_pc)
    # add associated training label column to PCA model DataFrame
    train_pc_df['y'] = ytrain

    # show explained variance associated with principal components
    exp_var_matrix = pca_train.explained_variance_ratio_
    print('Explained variation per PC: {}'.format(exp_var_matrix))
    print('Total explained variation by PCs: {}%'.format(round(exp_var_matrix.sum() * 100, 2)))

    # plot scatterplot of PCA model (2 PCs)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x='PC 1', y='PC 2', hue="y",
                    palette=sns.color_palette("hls", 10), data=train_pc_df, legend="full", alpha=0.3)
    plt.show()

    end = time.time()  # program end time
    print("Execution Time: {} seconds".format(round(end - begin, 2)))
