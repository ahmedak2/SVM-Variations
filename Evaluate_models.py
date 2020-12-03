import time


def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.__init__() # reinitialize model
    t_prev = time.time()
    model.train(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = ((y_test==y_pred).sum())/float(y_test.shape[0])
    comp_time = time.time() - t_prev
    
    return comp_time, acc

def evaluate_using_mnist(model):
    # load odd/even data first
    import Mnist_loader as mnist
    x_train, y_train, x_test, y_test = mnist.load_odd_even_Mnist()
    x_train, x_test, mu, std = mnist.preprocess_Mnist(x_train, x_test)
    
    comp_time_oe, acc_oe = evaluate_model(model, x_train, y_train, x_test, y_test)
    
    # load 3/8 data:
    x_train, y_train, x_test, y_test = mnist.load_two_numbers_Mnist()
    x_train, x_test, mu, std = mnist.preprocess_Mnist(x_train, x_test)
    comp_time_38, acc_38 = evaluate_model(model, x_train, y_train, x_test, y_test)
    
    return comp_time_oe, acc_oe, comp_time_38, acc_38
    
def evaluate_using_SVM_Guide1(model, use_colab = False, path = ''):
    from SVMGuide_loader import load_SVMGuide1
    x_train, y_train, x_test, y_test, mu, std = load_SVMGuide1(use_colab, path)
    comp_time, acc = evaluate_model(model, x_train, y_train, x_test, y_test)
    
    return comp_time, acc