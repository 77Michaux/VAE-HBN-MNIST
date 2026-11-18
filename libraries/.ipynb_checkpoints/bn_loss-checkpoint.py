import numpy as np
import pandas as pd

def to_disc(val, select) :
    for s, sel in enumerate(select) :
        if val <= sel :
            return s
    return len(select) -1

def discretize_MorphoMNIST(verified_MorphoMNIST_cont, intervals) :    
    colnames = intervals.columns
    tilde_y_disc_verified = pd.DataFrame()
    for label in colnames :
        tilde_y_disc_verified[label] = verified_MorphoMNIST_cont[label].apply((lambda val : to_disc(val, intervals.loc[1:, label])))

    return tilde_y_disc_verified


def dist_to_mod(measure, belief, interval) :
    belief = int(belief)
    real_mod = int(to_disc(measure, interval[1:]))
    
    if real_mod == belief :
        return 0
    if real_mod < belief :
        return interval[belief] -measure
    return measure -interval[belief +1]


def weighted_dist_to_mod(measure, belief_mod, interval) :
    """
    This function compute an the distance between the real measure of the descriptor and its hypothetic modality. 
    Here the center of its associated interval
    All interval have not the same distance, to avoid an overwheigthing, 
    we decide to normalise the value of each error within an interval by its own range.
    """
    
    belief_mod = int(belief_mod)
    verif_mod = int(to_disc(measure, interval[1:]))

    interval_range = interval[belief_mod +1] - interval[belief_mod]
    if belief_mod == verif_mod : return pd.Series([np.abs((interval[belief_mod +1] + interval[belief_mod])/2 -measure)/interval_range])
       
    # if not the same modality => dist >= 0.5
    dist = 0.5 
    
    # distance between the measure and the bound of its own interval
    mod_gap = belief_mod - verif_mod
    measure_interval_range = interval[verif_mod +1] -interval[verif_mod] # normalise verif mod error
    measure_dist = interval[verif_mod +1] -measure if mod_gap > 0 else measure -interval[verif_mod]

    # if the modality gap is 2 or mor that implies that one or more intervals are traversed 
    return pd.Series([dist +measure_dist/measure_interval_range + np.abs(mod_gap) -1]) # belief_mod, verif_mod, measure,


def measure_distance_to_modalities(funct, verified_MorphoMNIST_cont, tilde_y_disc, intervals) :
    colnames = intervals.columns
    tilde_y_disc_distance = pd.DataFrame()
    for label in colnames :
        cur_df = pd.DataFrame()
        cur_df["measure"] = verified_MorphoMNIST_cont[label]
        cur_df["belief"] = tilde_y_disc[label]
        tilde_y_disc_distance[label] = cur_df.apply((lambda val : funct(val.measure, val.belief, list(intervals.loc[:, label]))), axis = 1)
    return tilde_y_disc_distance








