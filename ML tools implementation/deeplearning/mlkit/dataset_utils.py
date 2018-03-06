import numpy as np 
import glob
from sklearn.utils import shuffle
import pandas as pd
import utils as ut
from sklearn.datasets import load_digits
from skimage import img_as_float
from skimage.io import imread
import image_utils as iu

def load_dataset(name, as_image=False):
    if name == "mnist_small":
        return ut.load_pkl("datasets/mnist_small.pickle")

    if name == "cifar_small":
        return ut.load_pkl("datasets/cifar_small.pickle")

    if name == "digits":
        digits = load_digits()
        X = digits["data"]
        y = digits["target"]
        X /= 255.

        X, y = shuffle(X, y)

        Xtest = X[500:]
        ytest = y[500:]

        X = X[:500]
        y = y[:500]

        return {"X":X, "y":y, "Xtest":Xtest, "ytest":ytest}
    if name == "boat_images":
        # LOAD SMALL IMAGES
        imgList = glob.glob("datasets/boat_images/*.png")

        df = pd.read_csv("datasets/boat_images/coords.csv")



        X = np.zeros((len(imgList), 80, 80, 3))
        Y = np.zeros((len(imgList), 80, 80))
        for i, img in enumerate(imgList):
            X[i] = img_as_float(imread(img))

            flag = False
            yx_coors = []
            for _, row in df.iterrows():
                if img[img.rindex("/")+1:] == row.image[row.image.rindex("/")+1:]:
                    yx_coors += [(row.y, row.x)]
                    flag = True
            if flag == False:
                Y[i] = np.zeros((80, 80))
            else:
                #Y[i] =  np.ones((80, 80))*-1

                for y, x in yx_coors:
                    Y[i, y, x] = 1

        X = iu.last2first(X)
        Y = Y[:, np.newaxis]

        if as_image:
            return X, Y
        else:
            y = Y.sum(axis=1).sum(axis=1).sum(axis=1)
       
            return X, y.astype(int)


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y