from LinearSVM import *
from Kmeans import *

import torch
import numpy as np

class KmeanSVM(object):
    def __init__(self, K=4, lamb=1e-1):
        self.Wl = None
        self.centroid = None
        self.K = K
        self.LSVM = LinearSVM()

        np.random.seed(0)
        torch.manual_seed(0)
        self.default_lr = 1
        self.default_reg = 1e-5

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

        I_Nx1 = torch.ones(N,1, dtype=X.dtype, device=X.device)
        X_with_ones = torch.cat((X,I_Nx1),dim=1)

        D = X_with_ones.shape[1]
        X_hat = torch.zeros(N, D*K, dtype=X.dtype, device=X.device)

        for l in range(K):
            idl = cluster_id == l
            X_hat[idl, (D*l):(D*(l+1))] = X_with_ones[idl,:]

        return X_hat


    def train(self, X_train, y_train, learning_rate=1, reg=1e-5, num_iters=100, batch_size=200, print_progress=False):
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
        W_hat = self.LSVM.W.clone().detach()
        
        # local predictor
        self.Wl = torch.zeros(D, self.K, dtype=X_train.dtype, device=X_train.device)
        for l in range(self.K):
            self.Wl[:,l] = W_hat[(D*l):(D*(l+1))]

        return loss_hist
    
    def predict(self, X_test):
        N = X_test.shape[0]
        #print("X_test", X_test.shape, "self.centroid", self.centroid.shape)
        # clustering
        cluster_label = Kmeans_testdata(X_test, self.centroid)

        I_Nx1 = torch.ones(N,1,dtype=X_test.dtype, device=X_test.device)
        X_test_with_ones = torch.cat((X_test,I_Nx1),dim=1)

        y_pred = torch.zeros(X_test.shape[0], dtype=X_test.dtype, device=X_test.device)
        for l in range(self.K):
            idl = cluster_label == l
            t = torch.matmul(X_test_with_ones[idl,:], self.Wl[:,l])
            y_pred[idl] = (1.0*(t>=0) -1.0*(t<0)).type(X_test.dtype)

        return y_pred
    
    def plot_classifier(self, plot_range):
        """
        Note: only for D=2 now!!
        Input:
        plot_range = [x_low, x_upper, y_low, y_upper] array like 
        """
        import matplotlib.pyplot as plt
        x = np.linspace(plot_range[0], plot_range[1], 50)
        y = np.linspace(plot_range[2], plot_range[3], 50)
        xx, yy = np.meshgrid(x, y)
        X_test = torch.tensor([[x1, x2] for x1, x2 in zip(np.ravel(xx), np.ravel(yy))], dtype=self.Wl.dtype, device=self.Wl.device)
        z_test = self.predict(X_test)
        z = np.reshape(z_test.cpu(), (xx.shape))
        h = plt.contourf(x, y, z, alpha=0.1)
        plt.show()

def search_kmsvm_hyperparams(model, x, y, folds, Ks, lambs, learning_rates, regs):
    """
        model to evaluate
        x: full training data NxD
        y: full training data labels N
        folds: number of folds for k fold cross validation
        learning_rates: list of learning rates to test
        regs: list of regularization terms to test (lambda)
        
        outputs: max_accuracy, max_learning_rate, max_reg
    """
    # divide data into training and validation sets:
    N = x.shape[0]

    # get random indices for dataset folds
    indices = torch.randperm(N)
    length = N // folds

    max_acc = 0.0
    max_K = 0.0
    max_lamb = 0.0
    max_reg = 0.0
    max_lr = 0.0
    for K in Ks:
        for lamb in lambs:
            for reg in regs:
                for lr in learning_rates:
            
                    acc = torch.zeros(folds)
                    for k in range(folds):
                        x_train = torch.cat((x[indices[0:k*length]],x[indices[(k+1)*length:]]),dim = 0)
                        y_train = torch.cat((y[indices[0:k*length]],y[indices[(k+1)*length:]]),dim = 0)
                        x_val = x[indices[k*length:(k+1)*length]]
                        y_val = y[indices[k*length:(k+1)*length]]

                        model.__init__(K=K, lamb=lamb)
                        model.train(x_train, y_train, reg=reg, num_iters=1000, learning_rate=lr, print_progress = False)
                        y_pred = model.predict(x_val)
                        acc[k] = ((y_val==y_pred).sum()) / float(y_val.shape[0])
                    acc_mean = acc.mean()
                    if(acc_mean > max_acc):
                        max_acc = acc_mean
                        max_K = K
                        max_lamb = lamb
                        max_reg = reg
                        max_lr = lr

                    print("at K = {}, lamb = {}, reg = {} and lr = {} we get acc = {}" .format(K, lamb, reg, lr, acc_mean))


    print("Max we get at: K = {}, lamb = {}, reg = {}, and lr = {} we get acc = {}" .format(max_K, max_lamb, max_reg, max_lr, max_acc))
    return max_acc, max_K, max_lamb, max_lr, max_reg


def apply_KMSVM(x_train, y_train, x_test=[], y_test=[], max_iters=1000, cross=False, K=4, lamb=1e-1):
    # get best learning rate and evaluate accuracy on test data
    # cross indicates whether we want to search for hyperparameters or just use the default ones
    
    if len(x_test) == 0:
        x_test = x_train
        y_test = y_train

    # apply cross validation if cross:
    if cross:
        Ks = [2,4,8,10]
        lambs = [1e-2, 1e-1, 1, 10, 20]
        learning_rates = [1e-1, 1, 10, 50, 100]
        regs = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        folds = 10

        KMSVM = KmeanSVM()
        max_acc, max_K, max_lamb, max_lr, max_reg = search_kmsvm_hyperparams(KMSVM, x_train, y_train, folds, Ks, lambs, learning_rates, regs)
        KMSVM = KmeanSVM(K=max_K, lamb=max_lamb)
        loss_history = KMSVM.train(x_train, y_train, reg=max_reg, num_iters=max_iters, learning_rate=max_lr, print_progress = False)
    else:
        KMSVM = KmeanSVM(K=K, lamb=lamb)
        loss_history = KMSVM.train(x_train, y_train, num_iters=max_iters, print_progress = False)
        
    y_pred = KMSVM.predict(x_test)
    acc = ((y_test==y_pred).sum())/float(y_test.shape[0])
    print(acc)
    return loss_history, KMSVM
