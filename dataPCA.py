import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# global constants
N = 2  # number of components for PCA


# load data from .npy file
def load_data(data):
    return np.load(data)


if __name__ == "__main__":
    # Load data
    xtrain = load_data('XtrDrivingImages.npy') / 255
    ytrain = load_data('YtrDrivingLabels.npy')
    # xtest = load_data('XteDrivingImages.npy') / 255
    # test_image_names = load_data('XteDrivingImageNames.npy')

    print("Min:{}, Max: {}".format(np.min(xtrain), np.max(xtrain)))

    xtrain_flat = xtrain.reshape(-1, 2304)

    print('Size of the flattened xtrain: {}'.format(xtrain_flat.shape))

    feat_cols = ['pixel' + str(i) for i in range(xtrain_flat.shape[1])]
    df_train = pd.DataFrame(xtrain_flat, columns=feat_cols)
    df_train['label'] = ytrain

    # print(df_train.head())  # show head of df_train (first 5 records)
    print('Size of the dataframe: {}'.format(df_train.shape))  # show size

    # initialize PCA model
    pca_train = PCA(n_components=N)

    # fit PCA model with DataFrame data from training matrix
    train_pc = pca_train.fit_transform(df_train.iloc[:, :-1])

    # create DataFrame from fit PCA model
    # train_pc_df = pd.DataFrame(data=train_pc, columns=['principal component 1', 'principal component 2'])
    n_pc = []
    for n in range(N):
        n_pc.append('principal component {}'.format(n + 1))

    # show principal component column labels
    print("principal component columns:\n{}".format(n_pc))
    train_pc_df = pd.DataFrame(data=train_pc, columns=n_pc)

    train_pc_df['y'] = ytrain  # add associated training label column to PCA model DataFrame
    print(train_pc_df.head())  # show head of PCA model DataFrame

    # show explained variance associated with each principal component
    exp_var_matrix = pca_train.explained_variance_ratio_

    print('Explained variation per principal component: {}'.format(exp_var_matrix))
    print('Total explained variation by principal components: {}%'.format(round(exp_var_matrix.sum() * 100, 2)))
