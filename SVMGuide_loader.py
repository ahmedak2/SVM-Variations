# Loading SVM Guide 1 Dataset:
import numpy as np
import torch
import os
from sklearn.datasets import load_svmlight_file

def load_SVMGuide1(USE_COLAB = False, path = ''):
    """
    Function that downloads data from the SVMGuide1 dataset. (numbers)
    The following are the inputs and outputs:

    Outputs:
        x_train: tensor of training data (3089,5)
        y_train: tensor of training data labels {-1,1}  (3089,)
        x_test: tensor of testing data (4000,5)
        y_test: tensor of testing data labels (4000,)
    """
    train_file = 'Data/SVMGuide1/svmguide1'
    test_file = 'Data/SVMGuide1/svmguide1.t'
    
    if USE_COLAB:
        train_file = os.path.join(path,train_file)
        test_file = os.path.join(path,test_file)

    # load training data
    train_data = load_svmlight_file(train_file)
    x_train_pre = torch.tensor(np.array(train_data[0].todense())).to(dtype = torch.float32, device = 'cpu')
    y_train = torch.tensor(np.array(train_data[1])).to(dtype = torch.int32, device = 'cpu')

    # load testing data
    test_data = load_svmlight_file(test_file)
    x_test_pre = torch.tensor(np.array(test_data[0].todense())).to(dtype = torch.float32, device = 'cpu')
    y_test = torch.tensor(np.array(test_data[1])).to(dtype = torch.int32, device = 'cpu')

    # scale data:
    mu = x_train_pre.mean(dim = 0)
    std = x_train_pre.std(dim = 0, unbiased = False)
    x_train_pre = (x_train_pre - mu) / (std + 1e-8)
    x_test_pre = (x_test_pre - mu) / (std + 1e-8)

    # add column of ones to 'x's
    train_ones = torch.ones_like(y_train)
    train_ones = train_ones.reshape([train_ones.shape[0], 1])
    x_train = torch.cat((x_train_pre, train_ones), dim = 1)

    test_ones = torch.ones_like(y_test)
    test_ones = test_ones.reshape([test_ones.shape[0], 1])
    x_test = torch.cat((x_test_pre, test_ones), dim = 1)

    # change y to {-1,1} instead of {0,1}
    y_train = y_train * 2 - 1
    y_test = y_test * 2 - 1
    
    return x_train, y_train, x_test, y_test, mu, std