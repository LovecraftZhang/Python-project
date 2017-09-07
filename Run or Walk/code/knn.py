"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        M, D = Xtest.shape
        I, J = self.X.shape
        k = self.k

        yhat = np.zeros(M)

        for i in range(M):
            p = Xtest[i]  # current point
            distance = np.zeros(I)   # array of distance from other points to the current point

            # calculate distance from p to every points in training set
            for n in range(I):
                distance[n] = np.linalg.norm(p - self.X[n])

            indices = np.argsort(distance)

            # pick first kth points and predict the yhat
            y = np.zeros(k)
            for j in range(k):
                y[j] = self.y[indices[j]]

            yhat[i] = utils.mode(y)

        return yhat


class CNN(KNN):

    def fit(self, X, y):

        Xcondensed = X[0:1,:]
        ycondensed = y[0:1]

        for i in range(1,len(X)):
            x_i = X[i:i+1,:]
            dist2 = utils.euclidean_dist_squared(Xcondensed, x_i)
            inds = np.argsort(dist2[:,0])
            yhat = utils.mode(ycondensed[inds[:min(self.k,len(Xcondensed))]])

            if yhat != y[i]:
                Xcondensed = np.append(Xcondensed, x_i, 0)
                ycondensed = np.append(ycondensed, y[i])

        self.X = Xcondensed
        self.y = ycondensed