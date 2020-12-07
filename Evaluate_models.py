import time

# C kernel svm 
# kmeans svm: K, lamb, learning_rate, reg
# CSVM: lamb K
def evaluate_model(model, x_train, y_train, x_test, y_test, C = None, K = None, lamb = None, learning_rate = None, reg = None):
    if K is not None: # Kmeans SVM!
        model.__init__(K = K, lamb = lamb) # reinitialize model
        t_prev = time.time()
        model.train(x_train, y_train, learning_rate = learning_rate, reg = reg)
        comp_time = time.time() - t_prev
    else:
        
        if C is not None: # our KSVM implementation!
            model.__init__(C = C) # reinitialize model
        else:
            model.__init__() # reinitialize model
            
        t_prev = time.time()
        model.train(x_train, y_train)
        comp_time = time.time() - t_prev
    
    y_pred = model.predict(x_test)
    acc = ((y_test==y_pred).sum())/float(y_test.shape[0])
    return comp_time, acc

def evaluate_using_mnist(model, use_colab = False, path = '', C = None, K = None, lamb = None, learning_rate = None, reg = None):
    # process parameter passed
    if C is None:
        C1 = None
        C2 = None
    else:
        C1 = C[0]
        C2 = C[1]
        
    if K is None:
        K1 = lamb1 = learning_rate1 = reg1 = None
        K2 = lamb2 = learning_rate2 = reg2 = None 
    else:
        K1 = K[0]
        lamb1 = lamb[0]
        learning_rate1 = learning_rate[0]
        reg1 = reg[0]
        K2 = K[1]
        lamb2 = lamb[1]
        learning_rate2 = learning_rate[1]
        reg2 = reg[1]
    
    # load odd/even data first
    import Mnist_loader as mnist
    x_train, y_train, x_test, y_test = mnist.load_odd_even_Mnist(USE_COLAB=use_colab, path=path)
    x_train, x_test, mu, std = mnist.preprocess_Mnist(x_train, x_test)
    
    
        
    comp_time_oe, acc_oe = evaluate_model(model, x_train, y_train, x_test, y_test, C = C1, K = K1, lamb = lamb1, learning_rate = learning_rate1, reg = reg1)
    
    # load 3/8 data:
    x_train, y_train, x_test, y_test = mnist.load_two_numbers_Mnist(USE_COLAB=use_colab, path=path)
    x_train, x_test, mu, std = mnist.preprocess_Mnist(x_train, x_test)
    comp_time_38, acc_38 = evaluate_model(model, x_train, y_train, x_test, y_test, C = C2, K = K2, lamb = lamb2, learning_rate = learning_rate2, reg = reg2)
    
    return comp_time_oe, acc_oe, comp_time_38, acc_38
    
def evaluate_using_SVM_Guide1(model, use_colab = False, path = '', C = None, K = None, lamb = None, learning_rate = None, reg = None):
    from SVMGuide_loader import load_SVMGuide1
    x_train, y_train, x_test, y_test, mu, std = load_SVMGuide1(use_colab, path)
    comp_time, acc = evaluate_model(model, x_train, y_train, x_test, y_test, C = C, K = K, lamb = lamb, learning_rate = learning_rate, reg = reg)
    
    return comp_time, acc

def evaluate_using_KDD(model, use_colab = False, path = '', C = None, K = None, lamb = None, learning_rate = None, reg = None):
    from KDD_loader import kdd_load
    x_train, y_train, x_test, y_test = kdd_load(use_colab, path)
    comp_time, acc = evaluate_model(model, x_train, y_train, x_test, y_test, C = C, K = K, lamb = lamb, learning_rate = learning_rate, reg = reg)
    
    return comp_time, acc