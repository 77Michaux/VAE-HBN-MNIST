import os
import numpy as np
import pandas as pd

import torch

"""

Pour verifier si les variables discretes associées au latents dans l'échantillonnage du BN, ilnous faut :
    - latent_samp : les latents échantillonnés
    - label_disc_samp : les variables discretes échantillonnées associées aux "latent_samp"
    - label_samp_inter : les intervalles continus associé aux "label_disc"  
    
    - label_cont_samp : la mesure réelle des images reconstruites à partir des "latent_samp" via Morpho-MNIST
    - samp_dist : une méthode de mesure de la distance entre les "label_cont_samp" et les "label_disc_samp" via les "label_samp_inter"

"""



def mode_to_df(modes, path) : 
    reps = []
    for mode in modes :
        path_mode = os.path.join(path, mode)
        for rep in os.listdir(path_mode) :
            reps.append(os.path.join(path, mode, rep)) # os.path.join(path_mode, rep)
    
    DF = []
    for path in reps :
        row = [] # path, mode, RT, alpha, beta, gamma, lambda
        mode = path.split("\\")[-2]
        parameters = path.split("\\")[-1]
        
        RT = 1 if "RT" in mode else 0
        p_dict = {"alpha": None, "beta": None, "gamma": None, "lambda": None} 
        for p in parameters.split("-") :
            p_name, p_value = p.split("_")
            p_dict[p_name] = float(p_value)

        row = [path, mode, RT] + list(p_dict.values())
        DF.append(row)
    return pd.DataFrame(np.array(DF), columns = ["path", "mode", "RT", "alpha", "beta", "gamma", "lambda"])


def get_dict_disc(morph_disc_path, morph_intervals_path) :
    """
    morph_disc_path : path of discrete categories
    morph_intervals_path : path of the same DataFrame with interval
    Get the dictionnary wich rely the number of each category and the quantitative interval of Hartemink disc
    """
    morph_disc = pd.read_csv(morph_disc_path, sep = ",", index_col=0)
    morph_intervals = pd.read_csv(morph_intervals_path, sep = ",", index_col=0)
    
    dict_disc = {}
    for c, col in enumerate(morph_disc.columns[:-1]) :
        columns = morph_intervals[[col]]
        columns[col +"disc"] = morph_disc[col]
        columns = columns.drop_duplicates().sort_values(by = [col +"disc"])
        dict_disc[col] = {i+1:inter for i, inter in enumerate(columns[col])}
        
    del morph_disc, morph_intervals
    return dict_disc

