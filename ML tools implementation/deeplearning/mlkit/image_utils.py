import pickle
import os
import sys
import numpy as np
import pylab as plt 

from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
from skimage.transform import rescale


def first2last(X):
    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1,2,0))
    if X.ndim == 4:
        return np.transpose(X, (0,2,3,1))

def last2first(X):
    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2,0,1))
    if X.ndim == 4:
        return np.transpose(X, (0,3,1,2))

def read_image(filename, rule="rgb", scale=None, shape=None):
    if rule == "rgb":
        img = imread(filename)

    if scale is not None:
        return rescale(img, scale)
    if shape is not None:
        return resize(img, shape)

    return img_as_float(img)



def single(img):
    img = np.squeeze(img)
    if img.ndim == 4:
        img = np.squeeze(img)

    if img.ndim == 3 and img.shape[0] == 3:
        plt.imshow(np.transpose(img, [1, 2, 0]))
    else:
        plt.imshow(img, cmap=plt.get_cmap('gray'))

def show(*imgList):
    N = len(imgList)
    if N == 1:
        single(imgList[0])
    else:

        for i, img in enumerate(imgList):
            if N > 3:
                plt.subplot(1 + N/3, 3, i+1)
            else:
                plt.subplot(N, 1, i+1)

            single(img)

    plt.tight_layout()
    plt.show() 

def orient(img):
    img = np.squeeze(img)
    if img.ndim == 4:
        img = np.squeeze(img)

    if img.ndim == 3 and img.shape[0] == 3:
       return np.transpose(img, [1, 2, 0])
    
    return img


def show_heat(model, img, colorbar=False, show=True, save=False):
    show_mask(img, model.predict(img), colorbar=colorbar, 
                   show=show, save=save)
    
def show_mask(img, mask, colorbar=False, show=True, save=False):
    img = orient(img)
    mask = orient(mask)

    plt.imshow(img); plt.imshow(mask, alpha=0.5)
    if colorbar:
        plt.colorbar()

    if show:
        plt.show()
        print

    if save:
        plt.savefig(save)
        print ("%s saved..." % save)


    
