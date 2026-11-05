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

def NLL(X: torch.Tensor, hat_X: torch.Tensor, reduction='sum') : # Negative Log Likelyhood for prediction 
    # F.log_softmax(model(data), dim=1)
    return F.nll_loss(hat_X, X, reduction='sum')

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
    # return (0.5 * (Z_cov**2).sum() - torch.diag(Z_cov).sum()) /Z.shape[0]
    return torch.abs(Z_cov -torch.diag(Z_cov)).sum() 

# DVAE Orthogonal loss
def ORT(Z) :
    return torch.norm(Z.T @ Z - torch.eye(Z.shape[1]), p='fro')

# DAE loss : mninimize covariance beetween C and S latent parts (C : content (ex : number label) ; S : style (ex : thickness))
def CS(C: torch.Tensor, S: torch.Tensor) : 
    cov_matrix = (C - C.mean(dim=0)).T @ (S - S.mean(dim=0))   # Covariance
    return torch.norm(cov_matrix, p='fro')

def Factor(Z, disc, permute_dims) : 
    B, D = Z.shape
    batch_perm = torch.randperm(B)
    Z_prim = Z[batch_perm[B//2:]].detach()
    Z = Z[batch_perm[:B//2]]
    
    # TC for VAE part
    D_Z = disc(Z)
    loss_Factor = BCE(torch.zeros(B//2, 1).view(-1, 1), D_Z) # -torch.mean(torch.log(D_Z +1e-8))  # 2* BCE(torch.zeros(B//2, 2).view(-1, 2), D_Z)
    
    # TC for DISC part
    D_Z_p = disc(Z_prim)
    Z_pperm = permute_dims(Z_prim)
    D_Z_pperm = disc(Z_pperm)
    
    # 2.a  Z independant - 2.b perm_Z dependant 
    loss_DISC = 0.5*(BCE(torch.ones(Z.shape[0], 1), D_Z_p)  + BCE(torch.zeros(Z.shape[0], 1), D_Z_pperm))
    return loss_Factor, loss_DISC
    

def MINE(Z: torch.Tensor, critic: nn.Module, nb_combine: int = 100) -> torch.Tensor:
    """
    Calcule la perte MINE pour minimiser la corrélation totale entre les dimensions latentes.

    Args:
        Z: Tensor des variables latentes de forme (B, D).
        critic: Réseau critique (MLP) estimant T_\theta(Z_i, Z_j).
        B: Taille du batch pour l'estimation des marginales.

    Returns:
        Loss MINE (scalaire) à minimiser pour découpler les dimensions.
    """
    B, D = Z.shape

    # find all possible combination except for i == j
    index = np.arange(D)
    index_combine = np.concatenate([np.array([index, [i]*D], dtype = int).T for i in index])
    index_combine_kept = index_combine[:, 0] != index_combine[:, 1]
    index_combine = index_combine[index_combine_kept]
    index_combine = index_combine[np.random.randint(0, len(index_combine), nb_combine)]

    # Prepare Z_joint for critic
    # Z_combine = torch.concatenate([torch.stack([liste, torch.Tensor([z]*D)]).T for z in Z[0]])
    Z_joint = torch.concatenate([Z[:, index_combine[:, 0]].unsqueeze(-1), Z[:, index_combine[:, 1]].unsqueeze(-1)], dim = 2)
    Z_joint = Z_joint.view(-1, 2)

    # Prepare Z_margin for critic
    Z_perm = Z[torch.randperm(B), ] # permute batch observations
    Z_marginal = torch.concatenate([Z[:, index_combine[:, 0]].unsqueeze(-1), Z_perm[:, index_combine[:, 1]].unsqueeze(-1)], dim = 2)
    Z_marginal = Z_marginal.view(-1, 2)
    

    # Estimation de l'information mutuelle
    T_joint = critic(Z_joint)  # Score pour les échantillons conjoints
    T_marginal = critic(Z_marginal)  # Score pour les échantillons marginaux

    
    # Perte MINE (borne inférieure de la MI)
    return T_joint.sum() /nb_combine, torch.log(torch.exp(T_marginal).sum()) /nb_combine 
    # On minimise la corrélation, donc on retourne VAE minimize MI, critic minimize -MI


    
    