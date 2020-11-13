"""
# code to import MNIST Dataset
# import using: import Mnist_loader

This file contains multiple functions that will be useful for the Mnist dataset.
The following are an overview of the functions. Go to them for details.

load_full_Mnist() : Loads the full mnist dataset.

load_sample_Mnist(num_train, num_test) : Loads num_train random training points 
                                         and num_test random testing points from Mnist dataset. 
                                         
plot_samples_Mnist(train_x, train_y) : plots 5 random images from train_x

load_odd_even_Mnist(num_train, num_test) : Loads num_train random training points and num_test random 
                                           testing points from Mnist dataset. With {-1,1} labels for odd or even numbers.
                                           

load_two_numbers_Mnist(num1, num2, num_train, num_test) : Loads num_train random training points and num_test random
                                                          testing points from Mnist that have either num1 or num2 labels.
                                                          With {-1,1} labels corresponding to num1, num2 respectively.

"""


import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_full_Mnist():
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


    train_x_full = torch.tensor(idx2numpy.convert_from_file(train_images_file))
    train_y_full = torch.tensor(idx2numpy.convert_from_file(train_labels_file))
    test_x_full = torch.tensor(idx2numpy.convert_from_file(test_images_file))
    test_y_full = torch.tensor(idx2numpy.convert_from_file(test_labels_file))

    return train_x_full, train_y_full, test_x_full, test_y_full

def load_sample_Mnist(num_train = 60000, num_test = 10000):
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
    train_x_full, train_y_full, test_x_full, test_y_full = load_full_Mnist()

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


def load_odd_even_Mnist(num_train = 60000, num_test = 10000):
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
    train_x, train_y, test_x, test_y = load_sample_Mnist(num_train, num_test)

    # convert the odd labels to -1
    train_y[train_y%2 == 1] = -1 
    test_y[test_y%2 == 1] = -1

    # convert the even labels to 1
    train_y[train_y%2 == 0] = 1 
    test_y[test_y%2 == 0] = 1 

    return train_x, train_y, test_x, test_y
    
def load_two_numbers_Mnist(num1=3, num2=8, num_train=6000, num_test=1000):
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
    train_x_full, train_y_full, test_x_full, test_y_full = load_full_Mnist()
    
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