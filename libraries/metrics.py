import numpy as np


def mse(x, y) :
    diff = x -y
    return (diff**2).mean(axis = (1, 2, 3))

def MSE(x, y) :
    err = mse(x, y)
    return err.mean(), err.std()

def PSNR(x, y, l = 1) :
    ALL_PSNR = 10*np.log10(l**2/mse(x, y))
    return ALL_PSNR.mean(), ALL_PSNR.std()

def SSIM(x, y, L = 1) :
    batchsize = x.shape[0]
    k1, k2 = 0.01, 0.03
    
    c1, c2 = k1 *L, k2 *L 
    c3 = c2/2
    
    mu_x, mu_y = x.mean(axis = (1, 2, 3)), y.mean(axis = (1, 2, 3))
    std_x, std_y = x.std(axis = (1, 2, 3)), y.std(axis = (1, 2, 3))
    cov_xy = ((x -mu_x.reshape((batchsize, 1, 1, 1)))* (y -mu_y.reshape((batchsize, 1, 1, 1)))).mean(axis = (1, 2, 3))
    
    L, C, S = (2 *mu_x *mu_y +c1)/(mu_x**2 + mu_y**2 +c1), (2 *std_x *std_y +c2)/(std_x**2 +std_y**2 +c2), (cov_xy +c3)/(std_x *std_y +c3)
    err = L*C*S
    return err.mean(), err.std()

def bpp(old_dims, new_dims, L = 8) :
    # CR = old_dims/new_dims
    return L *new_dims /old_dims # = L/CR

def inv_bpp(bpp_val, old_dims = 512**2, L = 8) :
    # CR = L/bpp_val
    return bpp_val *old_dims /L # = old_dims/CR


def D_KL(P, Q) : 
    return (P*np.log(P/Q)).sum()

def Jensen_Shannon(P, Q) :
    M = (P+Q)/2
    return (D_KL(P, M) +D_KL(Q, M)) /2 


def format_pow10(x):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return f"{mantissa:.2f}e{exponent}"

