from LinearSVM import *
from Kmeans import *

import torch
import numpy as np

import matplotlib.pyplot as plt 

class ClusterSVM(object):
    def __init__(self, K=8, lamb=1e30):
        self.W = None
        self.Wl = None
        self.centroid = None

        self.K = K
        self.lamb = lamb

        self.LSVM = LinearSVM()

        np.random.seed(0)
        torch.manual_seed(0)
        self.default_lr = 0.5
        self.default_reg = 1e-3

    def feature_extension(self, X, cluster_id):
        """
        Inputs:
        - X: A PyTorch tensor of shape (N,D) containing D features of N data.
        - cluster_id: a PyTorch tensor of shape (N,) containing cluster index for each data
        
        Outputs:
        - X_hat: extended feature vectors of shape (N,D*(1+K))
        """
        N = X.shape[0]
        K = self.K
        lamb = self.lamb

        I_Nx1 = torch.ones(N,1, dtype=X.dtype, device=X.device)
        X_with_ones = torch.cat((X,I_Nx1),dim=1)

        D = X_with_ones.shape[1]

        X_hat = torch.zeros(N,D*(K+1), dtype=X.dtype, device=X.device)
        X_hat[:,:D] = 1/np.sqrt(lamb)*X_with_ones

        for l in range(K):
            idl = cluster_id == l

            X_hat[idl, (D*(l+1)):(D*(l+2))] = X_with_ones[idl,:]

        return X_hat


    def train(self, X_train, y_train, learning_rate=0.5, reg=1e-3, num_iters=100, batch_size=200, print_progress=False):
        """
        Inputs:
        - X_train: A PyTorch tensor of shape (N, D) containing training data; there are N training samples each of dimension D.
        - y_train: A PyTorch tensor of shape (N,) containing training labels; y[i] = {-1,1} means that X[i] has label  -1 or 1 depending on the class.
        - K: number of clusters
        - lamb: global regularization factor
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength. (ie. lambda)
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - print_progress: (boolean) If true, print progress during optimization.
        - exit_diff: (float) condition to stop the gradient descent algorithm if the change in loss is too low.
        Returns: A tuple of:
        - loss_all: A PyTorch tensor giving the values of the loss at each training iteration.
        """
        N = X_train.shape[0]
        D = X_train.shape[1]+1

        # clustering
        cluster_label, centroid = Kmeans(X_train,self.K)

        self.centroid = centroid

        # feature extension
        X_train_hat = self.feature_extension(X_train, cluster_label)

        # train linear SVM
        loss_hist = self.LSVM.train(X_train_hat, y_train, reg=reg, num_iters=num_iters, learning_rate=learning_rate)

        # SVM parameters
        # W_hat = torch.tensor(self.LSVM.W, dtype=X_train.dtype, device=X_train.device)
        W_hat = self.LSVM.W.clone().detach()

        # global regularizer
        self.W = 1/np.sqrt(self.lamb)*W_hat[:D]

        # local predictor
        self.Wl = torch.zeros(D,self.K, dtype=X_train.dtype, device=X_train.device)
        for l in range(self.K):
            self.Wl[:,l] = W_hat[(D*(l+1)):(D*(l+2))] + self.W

        return loss_hist
    
    def predict(self, X_test):
        N = X_test.shape[0]

        # clustering
        cluster_label = Kmeans_testdata(X_test, self.centroid)


        I_Nx1 = torch.ones(N,1,dtype=X_test.dtype, device=X_test.device)
        X_test_with_ones = torch.cat((X_test,I_Nx1),dim=1)

        y_pred = torch.zeros(X_test.shape[0], dtype=X_test.dtype, device=X_test.device)
        for l in range(self.K):
            idl = cluster_label == l

            t = torch.matmul(X_test_with_ones[idl,:], self.Wl[:,l])

            y_pred[idl] = (1.0*(t>=0) -1.0*(t<0)).type(X_test.dtype)

        # feature extension
        # X_test_hat = self.feature_extension(X_test, cluster_label)
        # y_pred = self.LSVM.predict(X_test_hat)

        return y_pred

    def plot_classifier(self, plot_range):
        """
        Note: only for D=2 now!!
        Input:
        plot_range = [x_low, x_upper, y_low, y_upper] array like 
        """
        x = np.linspace(plot_range[0], plot_range[1], 50)
        y = np.linspace(plot_range[2], plot_range[3], 50)
        xx, yy = np.meshgrid(x, y)
        X_test = torch.tensor([[x1, x2] for x1, x2 in zip(np.ravel(xx), np.ravel(yy))], device=self.W.device, dtype=self.W.dtype)
        z_test = self.predict(X_test)
        z = np.reshape(z_test.cpu(), (xx.shape))
        h = plt.contourf(x, y, z, alpha=0.1)
        plt.show()