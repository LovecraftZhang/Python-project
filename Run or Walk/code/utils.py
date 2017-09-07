from scipy import stats
import numpy as np

def mode(y):
    """Computes the element with the maximum count"""

    if len(y) == 0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]

def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest' """

    return np.sum(X ** 2, axis=1)[:, None] + np.sum(Xtest ** 2, axis=1)[None] - 2 * np.dot(X, Xtest.T)