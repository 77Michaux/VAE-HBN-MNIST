import torch
import torch.nn as nn

# from libraries.model_parts.cnn import Encoder, Decoder
from libraries.model_parts.loss import MSE, BCE, KLD, COV, ORT, Factor, MINE, CS

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
        

## Disentangle VAE
class DVAE(nn.Module):
    def __init__(self, CNN, TCNN, D, F, FILTER = 25, k = 2):
        super(DVAE, self).__init__()
        self.F = F
        self.D = D
        self.vae = VAE_based(CNN, TCNN, D, F, FILTER, k = k)
    
    
    def forward(self, X, alpha = 0.5, beta = 1., gamma = 1.) :
        losses = self.vae(X, alpha, beta, gamma)
        Z = losses[-1]
        
        loss = losses[0]
        loss_items = losses[1:-1]
        loss_ORT = ORT(Z)
        
        loss = loss + gamma *loss_ORT
        return (loss,) + loss_items + (loss_ORT.item(),)

## Disentangle VAE
class CSVAE(nn.Module):
    def __init__(self, CNN, TCNN, D, F, FILTER = 25):
        super(DVAE, self).__init__()
        self.F = F
        self.D = D
        self.vae = VAE_based(CNN, TCNN, D, F, FILTER)

    def forward(self, X, alpha = 0.5, beta = 1., gamma = 1.) :
        losses = self.vae(X, alpha, beta, gamma)
        Z = losses[-1]
        C, S = torch.split(Z, self.D//2, 1)
        
        loss = losses[0]
        loss_items = losses[1:-1]
        loss_CS = CS(C, S)
        
        loss = loss + gamma *loss_CS
        return (loss,) + loss_items + (loss_CS.item(),)

        
## FactorVAE
# Discriminateur du FactorVAE :
class DISC(nn.Module):
    def __init__(self, D, F = 1000):
        super(DISC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features = D, out_features = F),
            nn.BatchNorm1d(F),
            nn.ReLU(),
            nn.Linear(in_features = F, out_features = F),
            nn.BatchNorm1d(F),
            nn.ReLU(),
            nn.Linear(in_features = F, out_features = 1),
            nn.Sigmoid(),
        )
    def forward(self, Z):
        return self.model(Z)
        
class FactorVAE(nn.Module):
    def __init__(self, CNN, TCNN, D, F, VAE_FILTER = 25, DISC_FILTER = 25, TC = "Factor"):
        super(FactorVAE, self).__init__()
        self.F = F
        self.TC = TC
        self.vae = VAE_based(CNN, TCNN, D, F, VAE_FILTER)
        self.disc = None
        if TC == "Factor" :
            self.disc = DISC(D, DISC_FILTER)
        elif TC == "MINE" :
            self.disc = DISC(2, DISC_FILTER)
    
    def permute_dims(self, z):
        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z.append(z_j[perm])
        return torch.cat(perm_z, 1)

    def forward(self, X, alpha = 0.5, beta = 1., gamma = 1., lamb = 0.1) :
        losses  = self.vae(X, alpha, beta, gamma)
        Z = losses[-1]
        loss_items = losses[1:-1]
        loss = losses[0]

        ratio_Factor = Z.shape.numel()
        
        loss_Factor, loss_DISC = None, None
        if self.TC == "Factor" :
            loss_COV = COV(Z)
            loss_Factor, loss_DISC = Factor(Z, self.disc, self.permute_dims)

            loss = loss + gamma *loss_COV
            loss = loss + lamb *loss_Factor
            return (loss, loss_DISC) + loss_items + (loss_COV.item()/ratio_Factor, loss_Factor.item()/ratio_Factor, loss_DISC.item()/ratio_Factor) 
        elif self.TC == "MINE" :
            T_joint, T_marginal = MINE(Z, self.disc)
            loss_Factor = T_joint -T_marginal
            
            T_joint_disc, T_marginal_disc = MINE(Z.detach(), self.disc)
            loss_DISC = -(T_joint_disc -T_marginal_disc)
            
            
            loss = loss + lamb *loss_Factor
            return (loss, loss_DISC) + loss_items + (loss_Factor.item(), ) 
            # loss, loss_DISC, loss.item(), loss_DISC.item(), loss_RECO.items(), loss_KLD.items(), loss_Factor.items()
        

