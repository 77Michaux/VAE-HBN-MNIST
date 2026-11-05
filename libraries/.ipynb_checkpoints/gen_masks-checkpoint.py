import numpy as np
import scipy.stats
import pandas as pd


def get_masks(X_STRUCT, beta = 2, q = 0.1, seed = 42) : 
    total_shape = X_STRUCT.shape 
    X_STRUCT = X_STRUCT.reshape((X_STRUCT.shape[0], ) + X_STRUCT.shape[-2:])
    indexes = np.array(np.where(X_STRUCT != 0))
    X_STRUCT[tuple(indexes)] = 1.
    
    indexes = pd.DataFrame(indexes.T, columns = ["N", "X", "Y"]).groupby("N")
    eff_per_group = list(indexes.count()["X"])
    
    
    MUS, STDS = [], []
    X_DOSE = []
    img_shape = total_shape[-1]
    x = np.arange(img_shape)
    np.random.seed(seed)
    for i, group in indexes : 
        idx = np.random.randint(eff_per_group[i])
        mu_x, mu_y = list(group.iloc[idx][["X", "Y"]])
        std_x, std_y = list((group[["X", "Y"]].max() - group[["X", "Y"]].min()) *(np.random.rand(2)/5 +1/5))
        mask = scipy.stats.gennorm.pdf(x, beta, mu_x, std_x).reshape((img_shape, 1)) @ scipy.stats.gennorm.pdf(x, beta, mu_y, std_y).reshape((1, img_shape))
    
        MUS.append([mu_x, mu_y])
        STDS.append([std_x, std_y])
        X_DOSE.append((1-q) * mask/mask.max() + q)
    
    return np.array(X_DOSE), np.array(MUS), np.array(STDS)





# def get_masks(X_STRUCT, beta = 2, q = 0.1, seed = 42) : 
#     total_shape = X_STRUCT.shape 
#     X_STRUCT = X_STRUCT.reshape((X_STRUCT.shape[0], ) + X_STRUCT.shape[-2:])
#     indexes = np.array(np.where(X_STRUCT != 0))
#     X_STRUCT[tuple(indexes)] = 1.
#     
#     indexes = pd.DataFrame(indexes.T, columns = ["N", "X", "Y"]).groupby("N")
#     eff_per_group = list(indexes.count()["X"])
#     
#     
#     MUS, STDS = [], []
#     X_DOSE = []
#     img_shape = total_shape[-1]
#     x = np.arange(img_shape)
#     np.random.seed(seed)
#     for i, group in indexes : 
#         idx = np.random.randint(eff_per_group[i])
#         mu_x, mu_y = list(group.iloc[idx][["X", "Y"]])
#         std_x, std_y = list((group[["X", "Y"]].max() - group[["X", "Y"]].min()) *(np.random.rand(2)/5 +1/5))
#         mask = scipy.stats.gennorm.pdf(x, beta, mu_x, std_x).reshape((img_shape, 1)) @ scipy.stats.gennorm.pdf(x, beta, mu_y, std_y).reshape((1, img_shape))
#     
#         MUS.append([mu_x, mu_y])
#         STDS.append([std_x, std_y])
#         X_DOSE.append((1-q) * mask/mask.max() + q)
#     
#     X_DOSE = X_STRUCT *np.array(X_DOSE, dtype = np.float32) 
#     X_STRUCT = np.concatenate([X_STRUCT.reshape(total_shape), X_DOSE.reshape(total_shape)], axis = 1, dtype = np.float32)
#     return X_STRUCT, np.array(MUS), np.array(STDS)

# def get_STRUCT_DOSE() :
    