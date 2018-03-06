import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as NeuralNet
import pandas as pd
from mlkit import dataset_utils as du
import matplotlib.pyplot as plt
from torch import nn
from sklearn.utils import shuffle
import models
from mlkit.torch_kit import base
from mlkit import image_utils as iu
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from mlkit import utils as ut 

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--part', required=True, 
                        choices=['1', '2', '3', '4', '5', '6', '7'])

    # io_args = parser.parse_args()
    part = '1' # io_args.part
    
    # LOAD DATASET

    if part == '1':
        data = du.load_dataset("digits")

        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        print(X[:10])
        quit()

        model = MLPClassifier(solver="adam", batch_size=100, hidden_layer_sizes=())
        
        results = []

        for i in range(500):
            model.partial_fit(X, y, classes=np.arange(10))       
        
            # EVALUATE TRAIN AND TEST CLASSIFICATION
            yhat = model.predict(X)
            trainscore = (yhat == y).mean()

            yhat = model.predict(Xtest)
            testscore = (yhat == ytest).mean()

            print("%d - Train score = %.3f" % (i, trainscore))
            print("%d - Test score = %.3f\n" % (i, testscore))

            results += [{"Train score": trainscore, "Test score":testscore}]
        
        results = pd.DataFrame(results)
        results.plot()
        plt.show()

    elif part == '2':
        data = du.load_dataset("digits")

        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']


        model = models.LinearModel(n_features=X.shape[1], 
                                   n_outputs=10)
        results = []

        for i in range(50):
            model.fit(X, y, epochs=10, batch_size=100, 
                      verbose=0, optimizer_name="adam", 
                      learning_rate=1e-3, weights_name="linaer")       
            
            # EVALUATE TRAIN AND TEST CLASSIFICATION
            yhat = np.argmax(model.predict(X), axis=1)
            trainscore = (yhat == y).mean()

            yhat = np.argmax(model.predict(Xtest), axis=1)
            testscore = (yhat == ytest).mean()

            print("%d - Train score = %.3f" % ((i+1)*10, trainscore))
            print("%d - Test score = %.3f\n" % ((i+1)*10, testscore))

            results += [{"Train score": trainscore, "Test score":testscore}]

        results = pd.DataFrame(results)
        results.plot()
        plt.show()

    elif part == '3':
        data = du.load_dataset("digits")

        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # CONVERT DATA TO IMAGES
        X = np.reshape(X, (-1,1,8,8))
        Xtest = np.reshape(Xtest, (-1,1,8,8))


        model = models.SmallCNNModel(n_channels=X.shape[1], 
                                img_dim=X.shape[2],
                                n_outputs=10)
        results = []

        for i in range(50):
            model.fit(X, y, epochs=10, batch_size=100, verbose=0)       
            
            # EVALUATE TRAIN AND TEST CLASSIFICATION
            yhat = np.argmax(model.predict(X), axis=1)
            trainscore = (yhat == y).mean()

            yhat = np.argmax(model.predict(Xtest), axis=1)
            testscore = (yhat == ytest).mean()

            print("%d - Train score = %.3f" % ((i+1)*10, trainscore))
            print("%d - Test score = %.3f\n" % ((i+1)*10, testscore))

            results += [{"Train score": trainscore, "Test score":testscore}]

        results = pd.DataFrame(results)
        results.plot()
        plt.show()

    elif part == '4':
        # FULLY CONNECTED NEURAL NETWORK ON MNIST
        data = du.load_dataset("mnist_small")

        X = data['X']
        y = np.argmax(data['y'], 1)
        Xtest = data['Xtest']
        ytest = np.argmax(data['ytest'], 1)

        model = models.LinearModel(n_features=X.shape[1], 
                                   n_outputs=10)

        results = []

        for i in range(10):
            model.fit(X, y, epochs=1, batch_size=50, verbose=0, weights_name="linear") # weight can be saved
        
            # EVALUATE TRAIN AND TEST CLASSIFICATION
            yhat = np.argmax(model.predict(X), axis=1)
            trainscore = (yhat == y).mean()

            yhat = np.argmax(model.predict(Xtest), axis=1)
            testscore = (yhat == ytest).mean()

            print("%d - Train score = %.3f" % (i, trainscore))
            print("%d - Test score = %.3f\n" % (i, testscore))

            results += [{"Train score": trainscore, "Test score":testscore}]

        results = pd.DataFrame(results)
        results.plot()
        plt.show()



        # BASELINE TEST score IS 91%

    elif part == '5':
        # CONVOLUTIONAL NEURAL NETWORK ON MNIST
        data = du.load_dataset("mnist_small")

        X = data['X']
        y = np.argmax(data['y'], 1)
        Xtest = data['Xtest']
        ytest = np.argmax(data['ytest'], 1)

        # CONVERT DATA TO IMAGES
        X = np.reshape(X, (-1,1,28,28))
        Xtest = np.reshape(Xtest, (-1,1,28,28))

        model = models.CNNModel(n_channels=X.shape[1], 
                                img_dim=X.shape[2],
                                n_outputs=10)

        results = {}

        for i in range(10):
            model.fit(X, y, epochs=1, batch_size=50, verbose=0)       
        
            # EVALUATE TRAIN AND TEST CLASSIFICATION
            yhat = np.argmax(model.predict(X), axis=1)
            trainscore = (yhat == y).mean()

            yhat = np.argmax(model.predict(Xtest), axis=1)
            testscore = (yhat == ytest).mean()

            print("%d - Train score = %.3f" % (i, trainscore))
            print("%d - Test score = %.3f\n" % (i, testscore))

        # BASELINE TEST score IS 91%

    elif part == '6':
        # CONVOLUTIONAL NEURAL NETWORK ON CIFAR
        data = du.load_dataset("cifar_small")

        X = data['X']
        y = np.argmax(data['y'], 1)
        Xtest = data['Xtest']
        ytest = np.argmax(data['ytest'], 1)

        model = models.CNNModel(n_channels=X.shape[1], 
                                img_dim=X.shape[2],
                                n_outputs=10)

        results = {}

        for i in range(10):
            model.fit(X, y, epochs=1, batch_size=50, verbose=0)       
        
            # EVALUATE TRAIN AND TEST CLASSIFICATION
            yhat = np.argmax(model.predict(X), axis=1)
            trainscore = (yhat == y).mean()

            yhat = np.argmax(model.predict(Xtest), axis=1)
            testscore = (yhat == ytest).mean()

            print("%d - Train score = %.3f" % (i, trainscore))
            print("%d - Test score = %.3f\n" % (i, testscore))

        # BASELINE TEST score IS 91%

    elif part == '7':
        X, y = du.load_dataset("boat_images", as_image=False)

        # TRAIN NETWORK
        model = models.AttentionModel(n_channels=3, n_outputs=1)
        model.fit(X, y, batch_size=23, epochs=100)
        show = lambda m, i: iu.show(m.get_heatmap(X)[i], X[i])
        import pdb; pdb.set_trace()  # breakpoint 387f960a //
        show(model, 1)
