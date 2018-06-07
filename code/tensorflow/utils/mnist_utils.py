import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display


# load MNIST dataset with added padding
def load_mnist_32x32():
    # mnist data is by default 28x28 so we add a padding to make it 32x32
    data = input_data.read_data_sets('MNIST_data', one_hot=False, reshape=False)
    # data cannot be directly modified because it has no set() attribute,
    # so we need to make a copy of it on other variables
    X_trn, y_trn = data.train.images, data.train.labels
    X_val, y_val = data.validation.images, data.validation.labels
    X_tst, y_tst = data.test.images, data.test.labels
    # we make sure that the sizes are correct
    assert(len(X_trn) == len(y_trn))
    assert(len(X_val) == len(y_val))
    assert(len(X_tst) == len(y_tst))
    # print info
    print("Training Set:   {} samples".format(len(X_trn)))
    print("Validation Set: {} samples".format(len(X_val)))
    print("Test Set:       {} samples".format(len(X_tst)))
    print("Labels: {}".format(np.unique(y_trn)))
    print("Original Image Shape: {}".format(X_trn[0].shape))
    # Pad images with 0s
    X_trn = np.pad(X_trn, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_val = np.pad(X_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_tst = np.pad(X_tst, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    print("Updated Image Shape: {}".format(X_trn[0].shape))
    
    # this is a trick to create an empty object,
    # which is shorter than creating a Class with a pass and so on...
    mnist = lambda:0
    mnist.train = lambda:0
    mnist.validation = lambda:0
    mnist.test = lambda:0
    # and we remake the structure as the original one
    mnist.train.images = X_trn
    mnist.validation.images = X_val
    mnist.test.images = X_tst
    mnist.train.labels = y_trn
    mnist.validation.labels = y_val
    mnist.test.labels = y_tst
    
    return mnist
