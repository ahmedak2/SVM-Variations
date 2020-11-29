# Mnist_loader.py
"""
# code to import MNIST Dataset
# import using: import Mnist_loader as mnist
# Then use mnist."function name" to call different functions

This file contains multiple functions that will be useful for the Mnist dataset.
The following are an overview of the functions. Go to them for details.

preprocess_Mnist(x_train, x_test) : Takes in Mnist images and returns flat arrays with ones column.

load_full_Mnist(USE_COLAB = False, path = '') : Loads the full mnist dataset.

load_sample_Mnist(num_train, num_test, USE_COLAB = False, path = '') : Loads num_train random training points 
                                         and num_test random testing points from Mnist dataset. 
                                         
plot_samples_Mnist(train_x, train_y, USE_COLAB = False, path = '') : plots 5 random images from train_x

load_odd_even_Mnist(num_train, num_test, USE_COLAB = False, path = '') : Loads num_train random training points and num_test random 
                                           testing points from Mnist dataset. With {-1,1} labels for odd or even numbers.
                                           

load_two_numbers_Mnist(num1, num2, num_train, num_test, USE_COLAB = False, path = '') : Loads num_train random training points and num_test random
                                                          testing points from Mnist that have either num1 or num2 labels.
                                                          With {-1,1} labels corresponding to num1, num2 respectively.

"""


import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
    

def preprocess_Mnist(x_train, x_test):
    """
    Function that flattens the Mnist images and adds column of ones to them
    """
    N1 = x_train.shape[0]
    x_train_temp = x_train.reshape([N1,-1])
    mu = x_train_temp.mean(dim=0)
    std = x_train_temp.std(dim=0, unbiased=False)
    x_train_temp = (x_train_temp - mu) /(std + 1e-8)
    temp_ones = torch.ones([N1,1], dtype = x_train.dtype, device = x_train.device)
    x_train_temp = torch.cat((x_train_temp, temp_ones), dim = 1)
    
    
    N2 = x_test.shape[0]
    x_test_temp = x_test.reshape([N2,-1])
    x_test_temp = (x_test_temp - mu) /(std + 1e-8)
    temp_ones = torch.ones([N2,1], dtype = x_test.dtype, device = x_test.device)
    x_test_temp = torch.cat((x_test_temp, temp_ones), dim = 1)
    
    return x_train_temp, x_test_temp, mu, std
    

def load_full_Mnist(USE_COLAB = False, path = ''):
    """
    Function that downloads data from the MNIST dataset. (numbers)
    The following are the inputs and outputs:

    Outputs:
        train_x_full: tensor of images sampled randomly from the Mnist training dataset. (60000,28,28)
        train_y_full: tensor of labels of those sampled training images. (60000,)
        test_x_full: tensor of images sampled randomly from the Mnist test dataset. (10000,28,28)
        test_y_full: tensor of labels of those sampled testing images. (10000,)
    """
    
    
    train_images_file = 'Data/Mnist/train-images.idx3-ubyte'
    test_images_file = 'Data/Mnist/t10k-images.idx3-ubyte'
    train_labels_file = 'Data/Mnist/train-labels.idx1-ubyte'
    test_labels_file = 'Data/Mnist/t10k-labels.idx1-ubyte'
    
    if USE_COLAB:
        train_images_file = os.path.join(path,'Data/Mnist/train-images.idx3-ubyte')
        test_images_file = os.path.join(path,'Data/Mnist/t10k-images.idx3-ubyte')
        train_labels_file = os.path.join(path,'Data/Mnist/train-labels.idx1-ubyte')
        test_labels_file = os.path.join(path,'Data/Mnist/t10k-labels.idx1-ubyte')
    


    train_x_full = torch.tensor(np.array(idx2numpy.convert_from_file(train_images_file))).to(dtype = torch.float32, device = 'cuda')
    train_y_full = torch.tensor(np.array(idx2numpy.convert_from_file(train_labels_file))).to(dtype = torch.int32, device = 'cuda')
    test_x_full = torch.tensor(np.array(idx2numpy.convert_from_file(test_images_file))).to(dtype = torch.float32, device = 'cuda')
    test_y_full = torch.tensor(np.array(idx2numpy.convert_from_file(test_labels_file))).to(dtype = torch.int32, device = 'cuda')

    return train_x_full, train_y_full, test_x_full, test_y_full

def load_sample_Mnist(num_train = 60000, num_test = 10000, USE_COLAB = False, path = ''):
    """
    Function that downloads sample of data from the MNIST dataset. (numbers)
    The following are the inputs and outputs:

    Inputs:
        num_train: number of desired training images.
        num_test: number of desired testing images.
    Outputs:
        train_x: tensor of images sampled randomly from the Mnist training dataset. (num_train,28,28)
        train_y: tensor of labels of those sampled training images. (num_train,)
        test_x: tensor of images sampled randomly from the Mnist test dataset. (num_test,28,28)
        test_y: tensor of labels of those sampled testing images. (num_test,)
    """
    # Make sure the number of samples required is within range
    if num_train > 60000:
        num_train = 60000
    elif num_train < 0:
        num_train = 1

    if num_test > 10000:
        num_test = 10000
    elif num_test < 0:
        num_test = 1

    # get the full Mnist dataset
    train_x_full, train_y_full, test_x_full, test_y_full = load_full_Mnist(USE_COLAB, path)

    temp_train_random_indices = np.random.permutation(train_x_full.shape[0]) # get random indices for all train set
    temp_test_random_indices = np.random.permutation(test_x_full.shape[0]) # get random indices for all test set

    train_indices = temp_train_random_indices[:num_train]
    test_indices = temp_test_random_indices[:num_test]

    # get num_train and num_test images and labels

    train_x = train_x_full[train_indices]
    train_y = train_y_full[train_indices]
    test_x = test_x_full[test_indices]
    test_y = test_y_full[test_indices]

    return train_x, train_y, test_x, test_y


def plot_samples_Mnist(train_x, train_y):
    """
    Function that shows a few random images from Mnist dataset.
    The following are the inputs and outputs:

    Inputs:
        x_train: tensor of training images from Mnist dataset
    """
    plt.figure(figsize=(12, 6))
    for i in range(5):
      plt.subplot(1, 5, i + 1)
      plt.imshow(train_x[i])
      plt.title(train_y[i].item())
      plt.axis('off')
    plt.gcf().tight_layout()


def load_odd_even_Mnist(num_train = 60000, num_test = 10000, USE_COLAB = False, path = ''):
    """
    Function that downloads sample of data from the MNIST dataset. (numbers)
    This defines odd or even labels instead of numbers. odd corresponds to y = -1, even corresponds to y = 1.
    The following are the inputs and outputs:

    Inputs:
        num_train: number of desired training images.
        num_test: number of desired testing images.
    Outputs:
        train_x: tensor of images sampled randomly from the Mnist training dataset. (num_train,28,28)
        train_y: tensor of labels of those sampled training images. This contains either odd or even labels only. (num_train,)
        test_x: tensor of images sampled randomly from the Mnist test dataset. (num_test,28,28)
        test_y: tensor of labels of those sampled testing images. This contains either odd or even labels only. (num_test,)
    """
    # load a random sample from Mnist dataset
    train_x, train_y, test_x, test_y = load_sample_Mnist(num_train, num_test, USE_COLAB, path)

    # convert the odd labels to -1
    train_y[train_y%2 == 1] = -1 
    test_y[test_y%2 == 1] = -1

    # convert the even labels to 1
    train_y[train_y%2 == 0] = 1 
    test_y[test_y%2 == 0] = 1 

    return train_x, train_y, test_x, test_y
    
def load_two_numbers_Mnist(num1=3, num2=8, num_train=6000, num_test=1000, USE_COLAB = False, path = ''):
    """
    Function that downloads sample data of 2 specific numbers from the MNIST dataset. (numbers)
    This labels are changed to be y = -1 corresponding to num1, and y = 1 corresponding to num2.
    The following are the inputs and outputs:

    Inputs:
        num_train: number of desired training images.
        num_test: number of desired testing images.
    Outputs:
        train_x: tensor of images sampled randomly from the Mnist training dataset. (num_train,28,28)
        train_y: tensor of labels of those sampled training images. This contains either odd or even labels only. (num_train,)
        test_x: tensor of images sampled randomly from the Mnist test dataset. (num_test,28,28)
        test_y: tensor of labels of those sampled testing images. This contains either odd or even labels only. (num_test,)
    """
    # get the full Mnist dataset
    train_x_full, train_y_full, test_x_full, test_y_full = load_full_Mnist(USE_COLAB, path)
    
    # get the data related to num1 and num2 in order
    train_x_nums = train_x_full[train_y_full == num1]
    train_y_nums = -torch.ones(train_x_nums.shape[0])
    train_x_nums = torch.cat((train_x_nums, train_x_full[train_y_full == num2]), dim = 0)
    train_y_nums = torch.cat((train_y_nums, torch.ones(train_x_full[train_y_full == num2].shape[0])), dim = 0)
    
    test_x_nums = test_x_full[test_y_full == num1]
    test_y_nums = -torch.ones(test_x_nums.shape[0])
    test_x_nums = torch.cat((test_x_nums, test_x_full[test_y_full == num2]), dim = 0)
    test_y_nums = torch.cat((test_y_nums, torch.ones(test_x_full[test_y_full == num2].shape[0])), dim = 0)
    
    # ensure num_train and num_test are within proper range
    if num_train > train_x_nums.shape[0]:
        num_train = train_x_nums.shape[0]
    elif num_train < 0:
        num_train = 1
    if num_test > test_x_nums.shape[0]:
        num_test = test_x_nums.shape[0]
    elif num_test < 0:
        num_test = 1
        
    temp_train_random_indices = np.random.permutation(train_x_nums.shape[0]) # get random indices for all train set of nums 1 and 2
    temp_test_random_indices = np.random.permutation(test_x_nums.shape[0]) # get random indices for all test set of nums 1 and 2

    train_indices = temp_train_random_indices[:num_train]
    test_indices = temp_test_random_indices[:num_test]

    # get random num_train and num_test images and labels from both num1 and num2
    train_x = train_x_nums[train_indices]
    train_y = train_y_nums[train_indices]
    test_x = test_x_nums[test_indices]
    test_y = test_y_nums[test_indices]
        
    
    return train_x, train_y, test_x, test_y