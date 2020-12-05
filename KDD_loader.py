import numpy as np
import torch
import os

def kdd_load(USE_COLAB = False, GOOGLE_DRIVE = ''):
    
    path = 'Data/KDD2012/track1/rec_log_train.txt'
    if USE_COLAB:
        path = os.path.join(GOOGLE_DRIVE,path)
    
    data = np.loadtxt(path, delimiter="\t")
    data = torch.tensor(data).to(dtype = torch.float32, device = 'cuda')
    
    # assign training and testing data. Last 10 million points are for testing
    N = 10000000
    x_train_pre = data[0:-N, 0:2]
    y_train = data[0:-N, 2]
    x_test_pre = data[-N:, 0:2]
    y_test = data[-N:, 2]
    
    
    # scaling data is meaningless for this dataset.
    
    # add column of ones to 'x's
    train_ones = torch.ones_like(y_train)
    train_ones = train_ones.reshape([train_ones.shape[0], 1])
    x_train = torch.cat((x_train_pre, train_ones), dim = 1)

    test_ones = torch.ones_like(y_test)
    test_ones = test_ones.reshape([test_ones.shape[0], 1])
    x_test = torch.cat((x_test_pre, test_ones), dim = 1)
    
    return x_train, y_train, x_test, y_test