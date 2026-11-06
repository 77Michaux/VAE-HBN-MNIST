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

    def forward(self, X, beta = 1., gamma = 1.) :
        hat_X = MU = LOGVAR = Z = None 
        F = X.shape[1]
        hat_X, MU, LOGVAR, Z = self.pred(X)

        loss_KLD = KLD(MU, LOGVAR)
        ratio_image = float(X.shape.numel()) /100
        loss_RECO = BCE(X, hat_X, "sum")
        loss = loss_RECO -beta*loss_KLD
        return loss, loss_RECO.item()/ratio_image, -loss_KLD.item()/ratio_image, Z
            
            

# VAE
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.vae = VAE_based(Encoder, Decoder)
        
    def forward(self, X, beta = 1.) :
        return self.vae(X, beta = beta)[:-1]


# TCVAE with Covariance loss
class TCVAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(TCVAE, self).__init__()
        self.vae = VAE_based(Encoder, Decoder)

    def forward(self, X, beta = 1., gamma = 1.) :
        losses  = self.vae(X, beta, gamma)
        Z = losses[-1]
        loss_items = losses[1:-1]
        loss = losses[0]

        ratio_COV = Z.shape[1]
        
        loss_COV = COV(Z)
        loss = loss + gamma *loss_COV
        return (loss, ) + loss_items + (loss_COV.item()/ratio_COV,) 
        # loss, loss.item(), loss_RECO.items(), loss_KLD.items(), loss_COV.items()
        

