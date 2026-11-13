import numpy as np
import pandas as pd

from matplotlib import pyplot as plt



#### FONCTIONS de SELECTION de VARIABLES ####
def rap_corr(y, x):
    '''
        x est une variable qualitative
        y est une variable quantitative
    '''
    somme = 0
    for facteur in x.value_counts().index:
        ind = x[x == facteur].dropna(axis=0).index
        classe = y.iloc[ind,]
        somme = somme + len(classe)*np.sum((np.mean(classe)-np.mean(y))**2)
    y_ecart = np.sum((y-np.mean(y))**2)
    rc = somme/y_ecart
    #rc.index = ['Rapport de Correlation']
    return rc

def rapport_de_correlation(df_latent, df_labels) :
    rc = [df_latent.apply(rap_corr, x = df_labels[col], axis = 0) for col in df_labels.columns]
    rc = pd.concat(rc, axis = 1)
    rc.columns = df_labels.columns
    return rc



def VAR_Inter_Intra(datas, labels, mode = "normal") :

    df = pd.DataFrame(datas, columns = list(range(1, datas.shape[-1] +1)))
    Eff = datas.copy()
    Mean_Eff = Eff.mean(axis = 0)

    df["label"] = labels
    groups = df.groupby("label")
    Eff_per_cat = [np.array(groups.get_group(x).iloc[:, :-1], dtype = np.float32) for x in groups.groups]
    nb_Eff_per_cat = np.array([len(x) for x in Eff_per_cat])
    nb_Eff_per_cat = nb_Eff_per_cat.reshape((len(nb_Eff_per_cat), 1))
    
    Mean_per_cat = np.array([np.array(cat.mean(axis = 0)) for cat in Eff_per_cat])
    Var_per_cat = np.array([np.array(cat.var(axis = 0)) for cat in Eff_per_cat])
    
    # Variance des moyenne : variance inter-classe
    var_inter = (Mean_per_cat-Mean_Eff)**2
    
    # Moyenne des variance : variance intra-classe
    var_intra = Var_per_cat
    
    Var_tot = Eff.var(axis = 0)*len(Eff)
    
    if mode == "detail" :
        return Var_tot, var_inter, var_intra
    
    return Var_tot, (nb_Eff_per_cat *var_inter).sum(axis = 0), (nb_Eff_per_cat *var_intra).sum(axis = 0) # df.apply(rap_corr, x = series, axis = 0)


# fonction stat per class
def get_stat_per_class(df_latent, mode = "mean") :
    
    if type(mode) == list :
        stats = []
        for m in mode :
            if m in ("mean", "std", "abs_mean") :
                stats.append(get_stat_per_class(df_latent, m))
        return tuple(stats)
    
    elif mode == "mean" :
        return df_latent.groupby(by = "LABEL").mean()
    
    elif mode == "std" :
        return df_latent.groupby(by = "LABEL").std()
    elif mode == "abs_mean" :
        df_latent = np.abs(df_latent)
        return df_latent.groupby(by = "LABEL").mean()
    

def plot_stat_per_class(stat1, stat2, axe_names = ["axe 1", "axe 2"], zoom = None) :

    
    for i, k in enumerate(stat1.index) :
        plt.scatter(stat1.loc[k, :], stat2.loc[k, :])

        for j, l in enumerate(stat1.columns):
            plt.annotate(f"{j} → {l}", (stat1.iloc[i, j], stat2.iloc[i, j]))

    plt.xlabel(axe_names[0])
    plt.ylabel(axe_names[1])
    plt.grid()
    plt.legend(stat1.index)
    
    if zoom is not None :
        x, y = zoom
        if x is not None :
            plt.xlim(x[0], x[1])
        if y is not None :
            plt.ylim(y[0], y[1])
    





