import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

## LOSS

def MSE(X: torch.Tensor, hat_X: torch.Tensor, reduction='sum'): # l'hypothèse du bruit entre X et hat_X est Gaussien 
    return F.mse_loss(X, hat_X, reduction='sum')

def BCE(X: torch.Tensor, hat_X: torch.Tensor, reduction='sum') : # l'hypothèse du bruit entre X et hat_X est Bernouilli
    return F.binary_cross_entropy(hat_X, X, reduction='sum')

# KL divergence VAE
def KLD(Z_MU: torch.Tensor, Z_LOGVAR: torch.Tensor) : # 
    return -1./2 * torch.sum(Z_MU.pow(2) + Z_LOGVAR.exp() -Z_LOGVAR -1)

# Total Corrélation
# def COV(Z) :
#     Z_cov = torch.cov(Z).T
#     return 0.5* torch.sum(Z_cov**2) - torch.sum(torch.diag(Z_cov))
def COV(Z):
    B = Z.shape[0]
    Z_centered = Z - Z.mean(dim=0)  # Centrage explicite
    Z_cov = (Z_centered.T @ Z_centered)/(B-1)  # Covariance non biaisée
    return torch.abs(Z_cov -torch.diag(Z_cov)).sum() 


    
    