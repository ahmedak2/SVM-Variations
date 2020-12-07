"""
Kernel SVM Class

kernel SVM by python:
https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/

QP solver qpth:
https://locuslab.github.io/qpth/
"""
import numpy as np
import torch
# import qpth
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
# import time

def linear_kernel(x1, x2):
    """
    x1 and x2 are np.array or torch.tensor with shape (D,) where D is the dimension.
    """
    return torch.dot(x1, x2)

def gaussian_kernel(x1, x2, gamma=1):
    """
    x1 and x2 are np.array or torch.tensor with shape (D,) where D is the dimension.
    """
    return torch.exp(-gamma * torch.norm(x1-x2)**2)


class kernelSVM(object):
    def __init__(self, kernel = gaussian_kernel, C=1e-2, eps=1e-5): 
        self.kernel = kernel
        if C is not None: # without slack variables
            self.C = float(C)
        else: # with slack variables
            self.C = C
        self.alpha = None
        self.sv = None
        self.sv_y = None
        self.b = 0.0
        self.eps = eps # no use for cvxpot
    
    def train(self, X, y, print_progress=False):
        """
        Take in the training data and labels and then save alpha, sv, sv_y, and b.
        X: A PyTorch tensor of shape (N, D) containing N data points and each point has dimension D.
        y: A PyTorch tensor of shape (N,) containing labels for the data.
        """
        N, D = X.shape
        #yy = np.array(y.cpu(), dtype = np.dtype(float))
        y = y.float()

        # Create kernel matrix K
        #t1 = time.time()
        K = torch.zeros((N, N),  device=X.device)
        for i in range(N):
            for j in range(N):
                if j>i:
                    kk = self.kernel(X[i,:], X[j,:])
                    K[i,j] = kk
                    K[j,i] = kk
                elif j==i:
                    K[i,j] = self.kernel(X[i,:], X[j,:])
        #t_k = time.time() - t1
        #print("t_k = ", t_k)
        print("start QP...")
        
        # Using qpth =========================
#         # Set up QP problem
#         Q = torch.ger(y, y) * K + self.eps*torch.eye(N, device=X.device) #torch.outer=torch.ger
#         p = -torch.ones(N, device=X.device)
#         A = torch.reshape(y, (1,N)) # reshape as 2D
#         b = torch.zeros(1, device=X.device)
        
#         if self.C is None:
#             G = torch.diag(-torch.ones(N, device=X.device))
#             h = torch.zeros(N, device=X.device)
#             #print("G", G.dtype, "h", h.dtype)
#         else:
#             G = torch.vstack((torch.diag(-torch.ones(N, device=X.device)), torch.eye(N, device=X.device)))
#             h = torch.hstack((torch.zeros(N, device=X.device), torch.ones(N, device=X.device)*self.C/N))
#             #print("G", G.dtype, "h", h.dtype)
        
#         # Solve alpha by QP
#         #t2 = time.time()
#         solution = qpth.qp.QPFunction(verbose=print_progress)(Q, p, G, h, A, b)
#         alpha = solution.view(-1) # reshape as 1D
#         #t_qp = time.time() - t2
#         #print("t_qp = ", t_qp)
        
        # Using cvxopt ======================
        # Set up QP problem
        K = np.array(K, dtype=np.float64)
        yy = np.array(y, dtype=np.float64)
        
        P = cvxopt.matrix(np.outer(yy, yy) * K)
        q = cvxopt.matrix(-np.ones(N))
        A = cvxopt.matrix(yy, (1,N)) # reshape as 2D
        b = cvxopt.matrix(0.0)
        #print(K[1:5,1:5],P[1:5,1:5])
        
        if self.C is None:
            G = cvxopt.matrix(np.diag(-np.ones(N)))
            h = cvxopt.matrix(np.zeros(N))
        else:
            G = cvxopt.matrix(np.vstack((np.diag(-np.ones(N)), np.identity(N))))
            h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N)*self.C/N)))
        
        # Solve alpha by QP
        cvxopt.solvers.options['show_progress'] = print_progress
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = torch.tensor(np.ravel(solution['x']))
        K = torch.tensor(K)
        # =======================================
        
        # Save support vectors
        isSV = alpha>1e-5
        idx = torch.arange(alpha.shape[0])[isSV]
        self.alpha = alpha[isSV]
        self.sv = X[isSV]
        self.sv_y = y[isSV]
        #print("%d support vectors out of %d points" % (len(self.alpha), N))
        
        # Calculate and save parameter b
        self.b = torch.sum(self.sv_y)
        for r in range(len(self.alpha)):
            self.b -= torch.sum(self.alpha * self.sv_y * K[idx[r], isSV])
        self.b = self.b / len(self.alpha)
    
    def predict(self, X):
        """
        Take in the test data and output a prediction torch.
        Input:
        -X: A PyTorch tensor of shape (N, D) containing N data points and each point has dimension D.
        Return:
        -y_pred: A PyTorch tensor of shape (N,) containing +1/-1 labels for the X
        """
        N, D = X.shape
        y_pred = torch.zeros(N, device=X.device)
        if self.kernel == linear_kernel:
            W = torch.zeros(D, device=X.device)
            for i in range(len(self.alpha)):
                W += self.alpha[i] * self.sv_y[i] * self.sv[i]
            print("X", X.dtype, "W", W.dtype)
            y_pred = torch.sign(torch.matmul(X, W) + self.b)
        else:
            for i in range(N):
                s = 0.0
                for alpha, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                    s += alpha * sv_y * self.kernel(X[i,:], sv)
                y_pred[i] = s
            y_pred = torch.sign(y_pred + self.b)
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
        X_test = torch.tensor([[x1, x2] for x1, x2 in zip(np.ravel(xx), np.ravel(yy))], device=self.sv.device, dtype=self.sv.dtype)
        z_test = self.predict(X_test)
        z = np.reshape(z_test.cpu(), (xx.shape))
        h = plt.contourf(x, y, z, alpha=0.1)
        plt.show()


def search_ksvm_hyperparams(model, x, y, kernel, C_list, eps_list, folds=3):#C_list, 
    """
        KSVM model to evaluate
        x: full training data NxD
        y: full training data labels N        
        outputs: max_accuracy, max_learning_rate, max_reg
    """
    # divide data into training and validation sets:
    N = x.shape[0]

    # get random indices for dataset folds
    indices = torch.randperm(N)
    length = N // folds

    max_acc = 0.0
    max_C = 0.0
    max_eps = 0.0
    for C in C_list:
        for eps in eps_list:
            acc = torch.zeros(folds)
            for k in range(folds):
                x_train = torch.cat((x[indices[0:k*length]],x[indices[(k+1)*length:]]),dim = 0)
                y_train = torch.cat((y[indices[0:k*length]],y[indices[(k+1)*length:]]),dim = 0)
                x_val = x[indices[k*length:(k+1)*length]]
                y_val = y[indices[k*length:(k+1)*length]]

                model.__init__(kernel, C, eps)
                model.train(x_train, y_train, print_progress = False)
                y_pred = model.predict(x_val)
                acc[k] = ((y_val==y_pred).sum()) / float(y_val.shape[0])
            acc_mean = acc.mean()
            if(acc_mean > max_acc):
                max_acc = acc_mean
                max_C = C
                max_eps = eps

            print("at C = {} and eps = {} we get acc = {}" .format(C, eps, acc_mean))


    print("Max we get at: C: {} and eps = {} we get acc = {}" .format(max_C, max_eps, max_acc))
    return max_acc, max_C, max_eps

def apply_KSVM(kernel, x_train, y_train, x_test = [], y_test = [], cross=False, C=None, eps=1e-5):
    # get best learning rate and evaluate accuracy on test data
    # cross indicates whether we want to search for hyperparameters or just use the default ones
    
    if len(x_test) == 0:
        x_test = x_train
        y_test = y_train

    # apply cross validation if cross:
    if cross:
        C_list = [1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3] # lambda=1/C
        eps_list = 1e-5 #[1e-6, 1e-5, 1e-4, 1e-3]
        KSVM = kernelSVM(kernel)
        max_acc, max_C, max_eps = search_ksvm_hyperparams(KSVM, x_train, y_train, kernel, C_list, eps_list)
        KSVM = kernelSVM(kernel, max_C, max_eps)
        KSVM.train(x_train, y_train, print_progress = False)
    else:
        KSVM = kernelSVM(kernel, C, eps)
        KSVM.train(x_train, y_train, print_progress = False)
    
    y_pred = KSVM.predict(x_test)
    acc = ((y_test==y_pred).sum())/float(y_test.shape[0])
    print("accuracy = ", float(acc))
    return KSVM


# function that uses library svm on input dataset and outputs accuracy
def apply_sklearn_ksvm(x_train, y_train, x_test = [], y_test = [], max_iters = 1000):
    
    if len(x_test) == 0:
        x_test = x_train
        y_test = y_train
    
    x_train = x_train.cpu()
    y_train = y_train.cpu()
    x_test = x_test.cpu()
    y_test = y_test.cpu()
    
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(x_train, y_train)
    y_pred = torch.tensor(svclassifier.predict(x_test))
    acc = ((y_test==y_pred).sum()) / float(y_test.shape[0])
    print(acc)


class KernelSVM_sklearn(object):
    def __init__(self):
        from sklearn.svm import SVC
        self.svc = SVC(kernel='rbf')
        print("init done")
    
    def train(self, x_train, y_train):
        self.svc.fit(x_train.cpu(), y_train.cpu())
        print("train done")
    
    def predict(self, x_test):
        y_pred = torch.tensor(self.svc.predict(x_test.cpu()), device=x_test.device)
        print("predict done")
        return y_pred

