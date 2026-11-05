import numpy as np


def load_datas(path, keys = ['x_train', 'y_train', 'x_test', 'y_test']):
    with np.load(path) as f:
        datas = [f[k] for k in keys]
        return datas