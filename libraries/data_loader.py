import numpy as np


def load_datas(path, keys = ['x_train', 'y_train', 'x_test', 'y_test']):
    with np.load(path) as f:
        datas = [f[k] for k in keys]
        return datas

def get_MNIST_analogie(X, Y, classe = 0, size = 200) : 
    class_idx = np.where(Y == classe)
    return X[class_idx][:size], Y[class_idx][:size]