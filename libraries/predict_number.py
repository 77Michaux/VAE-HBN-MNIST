import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from libraries.model_parts.loss import BCE

"""
Pour predire les nombres à partir des représentations latentes, il nous faut : 
    - net : le modèle utilisé
    - latent_vae : les latents issus du modèle "net" qui ont servi d'entrainement au réseau bayésien
    - latent_vae_val : des latents de validation issus du model "net" pour controler le surapprentissage du modèle
    - Y_train, Y_val : les nombre connus associés respectivmeents aux "latent_vae" et "latent_vae_val"

    - pred : un predicteur : 
        - Predicteur : le model
        - un entrainement de ce modèle
        - son enregistrement

    - latent_vae_bn : les latents echantillonné par le BN
    - Y_bn : les classes prétenduement associées au latent échantillonné par le BN

"""

class Predicter(nn.Module) :
    def __init__(self, input_dim) :
        super(Predicter, self).__init__()
        self.D = input_dim
        h1, h2 = 1000, 100
        self.model = nn.Sequential(
            nn.Linear(in_features=self.D, out_features=h1),
            nn.ReLU(),
            nn.Linear(in_features=h1, out_features=h2),
            nn.ReLU(),
            nn.Linear(in_features=h2, out_features=10),
            nn.Sigmoid(),
        )

    def to_one_hot(self, Y) :
        Y_one_hot = np.zeros((Y.shape[0], 10), dtype=np.float32)
        Y_one_hot[(np.arange(Y.shape[0]), Y)] = 1.0
        Y_one_hot = torch.from_numpy(Y_one_hot)
        return Y_one_hot

    def pred(self, Z) :
        one_hot_pred = self.model(Z).detach().numpy()
        return one_hot_pred.argmax(axis = 1)
        
    def forward(self, Z, Y) :
        hat_Y = self.pred(Z)
        Yoh = self.to_one_hot(Y)
        hat_Yoh = self.model(Z)
        
        loss_Yot = BCE(Yoh, hat_Yoh)
        loss_pred = (Y ==hat_Y).sum()
        return loss_Yot, loss_pred


def get_latents(model_path) : 
    latent_file = model_path + r"\latent.csv"
    latent_val_file = model_path + r"\latent_val.csv"
    latent_samp_file = model_path + r"\latent_samp.csv"

    if os.path.isfile(latent_file) and os.path.isfile(latent_val_file) and os.path.isfile(latent_samp_file) :
        latent = pd.read_csv(latent_file)
        latent_val = pd.read_csv(latent_val_file)
        latent_samp = pd.read_csv(latent_samp_file).dropna()
        Y_samp = latent_samp["number"].copy()
        latent_samp = latent_samp[[c for c in latent_samp.columns if c.isdigit()]]
        return latent, latent_val, latent_samp, Y_samp
    
    if not os.path.isfile(latent_file) : 
        print(f"{latent_file} does not exist")
        
    if not os.path.isfile(latent_val_file) : 
        print(f"{latent_val_file} does not exist")
        
    if not os.path.isfile(latent_samp_file) : 
        print(f"{latent_samp_file} does not exist")

def latent_shuffle(latents, Y_train, seed = 0) : 
      
    latent, latent_val, latent_samp, Y_samp = latents
    latent_shuf = np.concatenate([latent[:30000], latent_samp[:30000]])
    Y_train_shuf = np.concatenate([Y_train[:30000], Y_samp[:30000]])

    latent_samp_shuf = np.concatenate([latent[30000:], latent_samp[30000:]])
    Y_samp_shuf = np.concatenate([Y_train[30000:], Y_samp[30000:]])

    np.random.seed(seed)
    idx_lat = np.arange(len(latent_shuf))
    np.random.shuffle(idx_lat)
    
    return latent_shuf[idx_lat], Y_train_shuf[idx_lat], latent_val, latent_samp_shuf, Y_samp_shuf
    
    


def train_pred(model_path, Y_train, Y_val, epochs = 100, batchsize = 500, learning_rate = 1e-3, shuffle = None) :
    latents = get_latents(model_path)
    if latents is not None :
        latent, latent_val, latent_samp, Y_samp = latents
        if shuffle is not None : 
            latent, Y_train, latent_val, latent_samp, Y_samp = latent_shuffle(latents, Y_train, seed = shuffle)
        
        net = Predicter(latent.shape[-1])
        torch.manual_seed(42)
        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        batch_train = range(0, latent.shape[0] +1, batchsize) 
        batch_val = range(0, latent_val.shape[0] +1, batchsize)
        batch_samp = range(0, latent_samp.shape[0] +1, batchsize) 
        ratio = 100/batchsize

        
        AVG_loss = []
        AVG_loss_val = []
        AVG_loss_samp = []
        str_losses = ["loss", "val_loss", "samp_loss", "loss_pred", "val_loss_pred", "samp_loss_pred"]
        for epoch in range(epochs) :
            avg_loss = []
            avg_val_loss = []
            avg_samp_loss = []
            avg_loss_pred = []
            avg_val_loss_pred = []
            avg_samp_loss_pred = []
            for b, batch in enumerate(batch_train[:-1]) :
                Z = latent[batch:batch_train[b+1]]
                Y = Y_train[batch:batch_train[b+1]]
        
                Z = torch.from_numpy(np.array(Z, dtype = np.float32))
                loss, loss_pred = net(Z, Y)
                
                # Rétropropagation du gradient
                optimizer.zero_grad()
                loss.backward()
                # Descente de gradient (une itération)
                optimizer.step()
                
                avg_loss.append(loss.item()*ratio)
                avg_loss_pred.append(loss_pred.item()*ratio)
        
            for b, batch in enumerate(batch_val[:-1]) :
                Z = latent_val[batch:batch_train[b+1]]
                Y = Y_val[batch:batch_train[b+1]]
        
                Z = torch.from_numpy(np.array(Z, dtype = np.float32))
                val_loss, val_loss_pred = net(Z, Y)
                avg_val_loss.append(val_loss.item() *ratio)
                avg_val_loss_pred.append(val_loss_pred.item() *ratio)
        
            for b, batch in enumerate(batch_samp) :
                Z = latent_samp[batch:batch_train[b+1]]
                Y = Y_samp[batch:batch_train[b+1]]
        
                Z = torch.from_numpy(np.array(Z, dtype = np.float32))
                val_loss, val_loss_pred = net(Z, Y)
                avg_samp_loss.append(val_loss.item() *ratio)
                avg_samp_loss_pred.append(val_loss_pred.item() *ratio)
        
            
            
            losses = [ avg_loss, avg_val_loss, avg_samp_loss, avg_loss_pred, avg_val_loss_pred, avg_samp_loss_pred]
            print_losses = " - ".join([f"{strloss}: {np.round(np.mean(loss), 3)}" for strloss, loss in zip(str_losses, losses)])
            print(f"{epoch +1:03d} {print_losses}")
            # clear_output()
            AVG_loss.append([avg_loss, avg_loss_pred])
            AVG_loss_val.append([avg_val_loss, avg_val_loss_pred])
            AVG_loss_samp.append([avg_samp_loss, avg_samp_loss_pred])
    
        AVG_loss = np.concatenate(AVG_loss, axis = 1).T
        AVG_loss_val = np.concatenate(AVG_loss_val, axis = 1).T
        AVG_loss_samp = np.concatenate(AVG_loss_samp, axis = 1).T
        df = pd.DataFrame(AVG_loss, columns = [str_losses[0], str_losses[3]])
        df_val = pd.DataFrame(AVG_loss_val, columns = [str_losses[1], str_losses[4]])
        df_samp = pd.DataFrame(AVG_loss_samp, columns = [str_losses[2], str_losses[5]])
    
        predicter_rep = model_path + r"\number_predicter"
        if not os.path.isdir(predicter_rep) : 
            os.mkdir(predicter_rep)
        if shuffle is not None: 
            predicter_rep = model_path + r"\number_predicter" + f"\\{shuffle}"
            if not os.path.isdir(predicter_rep) : 
                os.mkdir(predicter_rep)
        predicter_file = predicter_rep + f"\\model.pt"
        torch.save(net, predicter_file)
    
        predicter_loss = predicter_rep + f"\\loss.csv"
        df.to_csv(predicter_loss, index = None)
        predicter_loss = predicter_rep + f"\\val_loss.csv"
        df_val.to_csv(predicter_loss, index = None)
        predicter_loss = predicter_rep + f"\\samp_loss.csv"
        df_samp.to_csv(predicter_loss, index = None)
        
        
        # return net, AVG_loss, AVG_val_loss, AVG_samp_loss, AVG_loss_pred, AVG_val_loss_pred, AVG_samp_loss_pred


























