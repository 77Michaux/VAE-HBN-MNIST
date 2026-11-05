import os

import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

from matplotlib import pyplot as plt

# Personal
import libraries.metrics as ms

def display_descriptor_cont_densities(save_on = False) :
    cont_path = r"02 - BN_Works/DATAS/continuous_morphomnist.csv"
    cont_morph = pd.read_csv(cont_path, sep = ";", index_col=None).iloc[:, :-1]
    
    w, h = 6, 1
    s = 3.5
    fig = plt.figure(figsize = (s*w, s*h))
    
    for i, col in enumerate(cont_morph.columns[:]) :
        Y = cont_morph[col]
        kde_y = gaussian_kde(Y)
        
        grid = np.linspace(Y.min(), Y.max(), 100) 
        p_Y = kde_y(grid)
    
        plt.subplot(1, w, i+1)
        plt.title(col.lower())
        plt.plot(grid, p_Y, linewidth = 2)
        plt.fill_between(
            x= grid, 
            y1= p_Y, 
            color= "b",
            alpha= 0.2)
    
        # plt.ylim(0, 1.0)
        # plt.grid(axis = "y")
    
    fig.text(0.5, -0.1, 'Densities', ha='center', fontsize = 14)
    fig.text(0.08, 0.5, 'Proportions', va='center', rotation='vertical', fontsize = 14)
    if save_on : 
        fig.savefig(os.path.join(path, "Morpho-MNIST - Density.pdf"), bbox_inches = 'tight')

def display_descriptor_disc_densities(model_path, disc_filename, save_on = False) :
    latent_samp_file = model_path + r"\latent_samp.csv"
    if os.path.isfile(latent_samp_file) :
        disc_path = r"02 - BN_Works/DATAS" + f"\\{disc_filename}"
        Y_real = pd.read_csv(disc_path, sep = ",", index_col=0)

        latent_samp = pd.read_csv(latent_samp_file)
        Y_gen = latent_samp[Y_real.columns]

        w, h = 4, 2
        fig = plt.figure(figsize = (3.5*w, 3.5*h))
        
        for i, col in enumerate(Y_real.columns) :
            plt.subplot(h, w, i +1)
            nb_mod = Y_real[col].unique()
            eff = len(Y_real)
        
            curr_Y_real = Y_real[[col]].value_counts().reset_index()
            curr_Y_gen = Y_gen[[col]].value_counts().reset_index()
            
            width = 0.15
            curr_Y_real["count"] = curr_Y_real["count"]/len(Y_real)
            curr_Y_gen["count"] = curr_Y_gen["count"]/len(Y_gen)
            JS = ms.Jensen_Shannon(curr_Y_real["count"], curr_Y_gen["count"])
            
            plt.bar(curr_Y_real[col] -width, curr_Y_real["count"], width = 2*width)
            plt.bar(curr_Y_gen[col] +width, curr_Y_gen["count"], width = 2*width)
        
            plt.title(f'{col.upper()} - JS = {ms.format_pow10(JS)}', fontsize = 11)
            plt.ylim(0, curr_Y_real["count"].max() *1.1)
            plt.grid(axis = "y")
            
            modality = sorted(curr_Y_real[col].unique())
            plt.xticks(modality)
                
        fig.text(0.5, 0.04, 'Modalities', ha='center', fontsize = 14)
        fig.text(0.04, 0.5, 'Proportions', va='center', rotation='vertical', fontsize = 14)
        if save_on : 
            fig.savefig(os.path.join(path, "Y_disc - Global dicrete distrib.pdf"))


def display_latent_densities(model_path, save_on = False) :
    latent_file = model_path + r"\latent.csv"
    latent_samp_file = model_path + r"\latent_samp.csv"
    if os.path.isfile(latent_file) and os.path.isfile(latent_samp_file) :
        
        latent = pd.read_csv(latent_file)
        latent_samp = pd.read_csv(latent_samp_file).dropna()
        latent_samp = latent_samp[[c for c in latent_samp.columns if c.isdigit()]]
        
        w, h = 4, 2
        fig = plt.figure(figsize = (5*w, 5*h))
        
        for i, col in enumerate(latent.columns[:8]) :
            
            plt.subplot(h, w, i +1)
            X, hat_X = latent[col], latent_samp[col]
            kde_real = gaussian_kde(X)
            kde_gen = gaussian_kde(hat_X)
            xmin = min(X.min(), hat_X.min())
            xmax = max(X.max(), hat_X.max())
            
            grid = np.linspace(xmin, xmax, 100) # grids = [np.linspace(xmin, xmax, n) for n in [200, 500, 1000, 2000, 5000]]
        
            p_X = kde_real(grid)
            p_hat_X = kde_gen(grid)
        
            JS = np.round(jensenshannon(p_X, p_hat_X), 5)
            plt.plot(grid, p_X, linewidth = 2)
            plt.plot(grid, p_hat_X, linewidth = 2)
            plt.title(f'j = {col} - JS = {ms.format_pow10(JS)}')
            plt.legend(["Original", "Generated"])
        
        
        fig.text(0.5, 0.05, f'Densities', ha='center', fontsize = 14)
        fig.text(0.085, 0.5, 'Proportions', va='center', rotation='vertical', fontsize = 14)
        
        if save_on : 
            fig.savefig("Distrib countinuous.pdf")
