import csv
import numpy as np
import sys
import argparse
import os
import matplotlib.pyplot as plt
import utils

from knn import CNN
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.neural_network import MLPClassifier as NeuralNet


# convert the data from 'dataset.csv' to 2D matrix

reader = csv.reader(open('dataset.csv', "r"))
x = list(reader)
originalData = np.array(x)

data = originalData[1:, 3:].astype("float")

np.random.shuffle(data)

X_data = data[:, 1:]
y_data = data[:, :1].astype("int")



n, d = X_data.shape

X = X_data[: int(n / 2)]
y = y_data[: int(n / 2)]
X_valid = X_data[int(n / 2):]
y_valid = y_data[int(n / 2):]


if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    # io_args = parser.parse_args()
    question = '2'  # io_args.question
    if question == '1':
        # data visualization(PCA)

        classes = np.array([['walking'], ['running']])

        pca = PCA(n_components=2)
        X_r = pca.fit(X_data).transform(X_data)

        # variance explained
        print('variance explained: %s' % str(pca.explained_variance_ratio_))

        plt.figure()
        colors = ['red', 'blue']
        lw = 0

        for color, i in zip(colors, [0, 1]):
            temp = y_data.reshape(n,)
            plt.scatter(X_r[temp == i, 0], X_r[temp == i, 1], color=color, alpha=.8, lw=lw,
                        label=classes[i])

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of Walk or Run Dataset')

        figname = os.path.join("..","figs", "visualization1.pdf")
        print("Saving", figname)
        plt.savefig(figname)

    if question == '2':
        # ISOMAP

        n_neighbors = 10
        n_components = 2

        Z = manifold.Isomap(n_neighbors, n_components).fit_transform(X)

        colors = ['yellow', 'blue']
        classes = np.array([['walking'], ['running']])

        plt.scatter(Z[:, 0], Z[:, 1], c=colors, label=classes)

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('ISOMAP of Walk or Run Dataset')

        figname = os.path.join("..", "figs", "visualization2.pdf")
        print("Saving", figname)
        plt.savefig(figname)





    if question == '3':
        # KNN model

        # train by CNN

        k = 20

        model_CNN = CNN(k)
        model_CNN.fit(X, y)
        y_CNN_training = model_CNN.predict(X)
        error_CNN_training = np.mean(y_CNN_training != y)
        print("CNN training error: ", error_CNN_training)

        # test on CNN
        y_CNN = model_CNN.predict(X_valid)
        error_CNN = np.mean(y_CNN != y_valid)
        print("CNN test error: ", error_CNN)
        Z, X = model_CNN.X.shape
        print("#variables in subset: ", Z)


    if question == '4':
        # neural network

        model = NeuralNet(
            solver="lbfgs",
            hidden_layer_sizes=(100,),
            early_stopping=True,
            activation="relu",
            alpha=0.001)

        model.fit(X, y)

        # Comput training error
        yhat = model.predict(X).reshape(int(n / 2), 1)
        trainError = yhat == y
        print("Training error = ", 1 - np.count_nonzero(trainError) / (n / 2))

        # Compute test error
        yhat = model.predict(X_valid).reshape(int(n / 2), 1)
        testError = yhat == y_valid

        print("Test error = ", 1 - np.count_nonzero(testError) / (n / 2))





