import os

import numpy as np
import pandas as pd

import torch
# from IPython.display import clear_output


def get_RT_X(X_STRUCT, X_DOSE) :
    X_STRUCT = X_STRUCT.astype(np.float32)
    X_STRUCT[np.where(X_STRUCT > 0)] = 1.
    
    X_DOSE = X_DOSE.astype(np.float32) /255
    X_DOSE = X_STRUCT*X_DOSE.reshape(X_STRUCT.shape)
    return np.concatenate([X_STRUCT, X_DOSE], axis = 1)

def load_MU_LOGVAR(net, X_train, M, F = 1) :
    batchsize = 500
    batch_range_train = range(0, X_train.shape[0] +1, batchsize) # 
    
    MU = []
    LOGVAR = []
    for i, batch in enumerate(batch_range_train[:-1]) :
        if i%10 == 9 : print(f"Step {i +1}/{len(batch_range_train) -1}")
        X = X_train[batch:batch_range_train[i +1]]
        if F == 1 :
            X = X.astype(np.float32)/255.
        elif F == 2 :
            X = get_RT_X(X, M[batch:batch_range_train[i +1]])
        X = torch.from_numpy(X)
        
        Z_MU, Z_LOGVAR = net.vae.Encoder(X)
        MU.append(Z_MU.detach().numpy())
        LOGVAR.append(Z_LOGVAR.detach().numpy())
    
        # clear_output()
    return np.concatenate(MU), np.concatenate(LOGVAR)

def get_Asquare(MU, VAR) :
    VAR_MEAN = np.var(MU, axis = 0)
    MEAN_VAR = np.mean(VAR, axis = 0)
    VAR_MEAN.shape, MEAN_VAR.shape
    
    Asquare = pd.DataFrame(VAR_MEAN/(VAR_MEAN+MEAN_VAR))
    Asquare = Asquare.sort_values(by = 0, ascending=  False).reset_index()
    return Asquare

# In the same path of net : Save (MU, LOGVAR), save MU_kept according thresh
    # load_MU_LOGVAR
    # get_Asquare
def save_latent_for_BN(model_path, X, M = None, thresh = 0.1, val = False) :
    model_file = os.path.join(model_path, "model.pt")
    if os.path.isfile(model_file) :
        net = torch.load(model_file, weights_only=False)

        latent_file = model_path + r"\latent.csv"
        latent_val_file =  model_path + r"\latent_val.csv"
        if not (val or os.path.isfile(latent_file)) :
            # get MU VAR
            print("Latent save...")
            MU, LOGVAR = load_MU_LOGVAR(net, X, M, F = net.F)
            np.savez(os.path.join(model_path, f"MU_LOGVAR"), MU = MU, LOGVAR = LOGVAR, allow_pickle=False)
            
            # Keep active vectors
            Asquare = get_Asquare(MU, np.exp(LOGVAR))
            idx_kept = Asquare[Asquare[0] >= thresh]["index"]
            
            # get a drawn Z
            MU_kept = MU[:, idx_kept]
            MU_kept = pd.DataFrame(MU_kept, columns = idx_kept)
            MU_kept.to_csv(latent_file, index = False)
            
            
        elif not os.path.isfile(latent_val_file) : 
            if os.path.isfile(latent_file) :
                latent = pd.read_csv(latent_file)
                print("Latent validation save...")
                MU, _ = load_MU_LOGVAR(net, X, M, F = net.F)
                idx_kept = np.array(latent.columns, dtype = np.uint8)
                
                MU_kept = MU[:, idx_kept]
                MU_kept = pd.DataFrame(MU_kept, columns = idx_kept)
                MU_kept.to_csv(latent_val_file, index = False)
            
            else :
                print("Need latent file before")
        else :
            print("Files do already exist")
            
def latent_to_Z(model_path, mode = "VAE") :
    latent_file = model_path + r"\latent.csv"
    if mode == "VAE" and os.path.isfile(latent_file) :
        latent = pd.read_csv(latent_file)
        Z = np.zeros((len(latent), 50), dtype = np.float32)
        Z[:, np.array(latent.columns, dtype = int)] = np.array(latent)
        return torch.from_numpy(Z)
        

    latent_file = model_path + r"\latent_samp.csv"
    if mode == "BN" and os.path.isfile(latent_file) :
        latent = pd.read_csv(latent_file).dropna()
        latent = latent[[c for c in latent.columns if c.isdigit()]]
        Z = np.zeros((len(latent), 50), dtype = np.float32)
        Z[:, np.array(latent.columns, dtype = int)] = np.array(latent)
        return torch.from_numpy(Z)



def Z_to_X(model_path, Z) :
    model_file = model_path + r"\model.pt"
    if os.path.isfile(model_file) :
        net = torch.load(model_file, weights_only=False)
        X = net.vae.Decoder(Z)
        return (X.detach().numpy() *255).astype(np.uint8)
         

def save_hat_X(model_path, mode = "VAE") :
    model_file = model_path + r"\model.pt"
    latent_file = model_path + r"\latent.csv" if mode == "VAE" else model_path + r"\latent_samp.csv"
    hat_X_file = model_path + r"\hat_X.npz" if mode == "VAE" else model_path + r"\X_samp.npz"
    if os.path.isfile(model_file) and os.path.isfile(latent_file) and not os.path.isfile(hat_X_file) :
        net = torch.load(model_file, weights_only=False)
        Z = latent_to_Z(model_path, mode = "VAE")
            
        batchsize = 1000
        batch_range_train = range(0, Z.shape[0] +1, batchsize)
        
        hat_X = []
        for b, batch in enumerate(batch_range_train[:-1]) :
            print(f"{b+1}/{len(batch_range_train) -1} -> ", end = "")
            z = Z[batch:batch_range_train[b +1]]
            hat_X.append(Z_to_X(model_path, z))
        hat_X = np.concatenate(hat_X).astype(np.uint8)
        np.savez(hat_X_file, hat_X = hat_X, allow_pickle = False)
    else : 
        if not os.path.isfile(model_file) : 
            print("Model does not exist")
        if not os.path.isfile(model_file) : 
            print("latent file does not exist")
        if os.path.isfile(hat_X_file) : 
            print(f"{hat_X_file.split("\\")[-1].split(".")[0]} file does exist")

def reparametrization_trick(MU, VAR) :
    SIGMA = torch.from_numpy(np.sqrt(VAR))
    MU = torch.from_numpy(MU)
    EPSILON = torch.empty_like(SIGMA).normal_()
    Z = MU +SIGMA*EPSILON
    return Z

def get_KLD(MU, VAR) :
    # (MU**2).mean(axis = 0)
    # -np.log(VAR).mean(axis = 0)

    KLD = (MU**2 + VAR -np.log(VAR) -1).mean(0)
    return KLD




# When the version of VAE change but not the Network
def refresh_save(models, paths) :
    for net, path in zip(models, paths) :
        check = torch.load(os.path.join(path, "model.pt"), weights_only=False)
        state_dict = check.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save(net, os.path.join(path, "model.pt"))