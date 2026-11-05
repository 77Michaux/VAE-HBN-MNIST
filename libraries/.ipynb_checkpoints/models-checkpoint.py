import torch
import torch.nn as nn

# from libraries.model_parts.cnn import Encoder, Decoder
from libraries.model_parts.loss import MSE, BCE, KLD, COV

# Base of VAE
class VAE_based(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE_based, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparametrize(self, MU, LOGVAR):
        if self.training:
            # the reparameterization trick
            SIGMA = LOGVAR.mul(0.5).exp_()
            ESPILON = torch.empty_like(SIGMA).normal_()
            return ESPILON.mul(SIGMA).add_(MU)
        else:
            return MU

    def E(self, Z) :
        k = self.k
        Z_cos = torch.cos(Z[:, :k].detach())
        Z_sin = torch.sin(Z[:, :k].detach())
        Z = torch.concatenate([Z_cos, Z_sin, Z[:,  k:]], axis = 1)
        return Z
    
    def pred(self, X) :
        MU, LOGVAR = self.Encoder(X)
        Z = self.reparametrize(MU, LOGVAR)
        hat_X = self.Decoder(Z)
        return hat_X, MU, LOGVAR, Z

    def forward(self, X, alpha_dose = 0.5, alpha_paroi = 0.5, beta = 1., gamma = 1.) :
        hat_X = MU = LOGVAR = Z = None 
        F = X.shape[1]
        if F <= 2 :
            hat_X, MU, LOGVAR, Z = self.pred(X)
        else : 
            hat_X, MU, LOGVAR, Z = self.pred(X[:, :]) # (0, 3)

        loss_KLD = KLD(MU, LOGVAR)
        ratio_image = float(X.shape.numel()) /100
        # ratio_latent = Z.shape.numel()
        
        if F == 1 :
            loss_RECO = BCE(X, hat_X, "sum")
            loss = loss_RECO -beta*loss_KLD
            return loss, loss_RECO.item()/ratio_image, -loss_KLD.item()/ratio_image, Z
            
        elif F == 2 :
            B = X.shape[0]            
            loss_DOSE_ORGAN = MSE(X[:B//2, 0], hat_X[:B//2, 0], "sum")
            loss_DOSE_PAROI = MSE(X[B//2:, 0], hat_X[B//2:, 0], "sum")
            
            loss_STRUCT_ORGAN = BCE(X[:B//2, 1], hat_X[:B//2, 1], "sum")
            loss_STRUCT_PAROI = BCE(X[B//2:, 1], hat_X[B//2:, 1], "sum")
            
            loss_ORGAN = alpha_dose*loss_DOSE_ORGAN +(1-alpha_dose)*loss_STRUCT_ORGAN
            loss_PAROI = alpha_dose*loss_DOSE_PAROI +(1-alpha_dose)*loss_STRUCT_PAROI
            loss = loss_ORGAN *(1-alpha_paroi) + loss_PAROI *alpha_paroi -beta *loss_KLD
            # loss = loss_DOSE_ORGAN -beta *loss_KLD
            return loss, loss_DOSE_ORGAN.item()/ratio_image, loss_STRUCT_ORGAN.item()/ratio_image, loss_DOSE_PAROI.item()/ratio_image, loss_STRUCT_PAROI.item()/ratio_image, -loss_KLD.item()/ratio_image, Z
            
        else : 
            loss_DOSE_ORGAN = MSE(X[:, 0], hat_X[:, 0], "sum")
            loss_DOSE_PAROI = MSE(X[:, 2], hat_X[:, 2], "sum")
            
            loss_STRUCT_ORGAN = BCE(X[:, 1], hat_X[:, 1], "sum")
            loss_STRUCT_PAROI = BCE(X[:, 3], hat_X[:, 3], "sum")
            
            loss_ORGAN = alpha_dose*loss_DOSE_ORGAN +(1-alpha_dose)*loss_STRUCT_ORGAN
            loss_PAROI = alpha_dose*loss_DOSE_PAROI +(1-alpha_dose)*loss_STRUCT_PAROI
            loss = loss_ORGAN *(1-alpha_paroi) + loss_PAROI *alpha_paroi -beta *loss_KLD
            # loss = loss_DOSE_ORGAN -beta *loss_KLD
            return loss, loss_DOSE_ORGAN.item()/ratio_image, loss_STRUCT_ORGAN.item()/ratio_image, loss_DOSE_PAROI.item()/ratio_image, loss_STRUCT_PAROI.item()/ratio_image, -loss_KLD.item()/ratio_image, Z
            

# VAE
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.vae = VAE_based(Encoder, Decoder)
        
    def forward(self, X, alpha_dose = 0.5, alpha_paroi = 0.5, beta = 1., gamma = 1.) :
        return self.vae(X, alpha_dose, alpha_paroi, beta, gamma)[:-1]

# TCVAE with Covariance loss
class TCVAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(TCVAE, self).__init__()
        self.vae = VAE_based(Encoder, Decoder)

    def forward(self, X, alpha_dose = 0.5, alpha_paroi = 0.5, beta = 1., gamma = 1.) :
        losses  = self.vae(X, alpha_dose, alpha_paroi, beta, gamma)
        Z = losses[-1]
        loss_items = losses[1:-1]
        loss = losses[0]

        ratio_COV = Z.shape[1]
        
        loss_COV = COV(Z)
        loss = loss + gamma *loss_COV
        return (loss, ) + loss_items + (loss_COV.item()/ratio_COV,) 
        # loss, loss.item(), loss_RECO.items(), loss_KLD.items(), loss_COV.items()
        

