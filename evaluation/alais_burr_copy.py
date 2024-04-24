#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
import random

from scipy.stats import norm
from .preprocessing import *
from .model import DNF
from .utils import *

VISIO_AMP = 1.
NB_RUNS = 50
X_MIN = -1.
X_MAX = 1.
DX = .02

# XP from Alais&Burr (2004)

df = pd.read_excel('dataVentriloquie.xlsx')
std_dict = calculate_std(df)
cond_dict = avg_std_par_condition(df)
print(std_dict)
#print(cond_dict)
#print(len(cond_dict)) # 512 correct
filtered_cond_dict = {key: value for key, value in cond_dict.items() if not np.isnan(value).any()}
#print(filtered_cond_dict)
#print(len(filtered_cond_dict))
#list of possible scenarios: [audio_pos, vis_pos, vis_std, expected_pos, expected_std, audio_rel, vis_rel]
_SCENARII = []
for key in filtered_cond_dict:
    audio_pos = key[2]
    vis_pos = key[3]
    expected_pos, expected_std = filtered_cond_dict[key]
    audio_rel = key[0]
    vis_rel = key[1]
    if (key[1], vis_pos) in std_dict:
        vis_std = std_dict[(key[1],vis_pos)]
    _SCENARII.append([audio_pos, vis_pos, vis_std, expected_pos, expected_std, audio_rel, vis_rel])

print(_SCENARII)
print("Number of scenarios: ", len(_SCENARII))

# function that calculates standard deviation for a certain intensity
def intensity_to_gaussian_param(df, cache, position_a, fiabilite_a):
    # Hay que usar dynamic programming
    if position_a in cache and fiabilite_a in cache[position_a]:
        return cache[position_a][fiabilite_a]
    # Las entradas de solo audio 
    filtered_df = df[(df['PositionV'] == 0) & (df['FiabiliteA'] == fiabilite_a) & (df['PositionA'] == position_a)]
    # desviacion de las posiciones estimadas
    std_a = filtered_df['X'].std()
    amplitude = 1 / (std_a * np.sqrt(2 * np.pi))

    if position_a not in cache:
        cache[position_a] = {}
    cache[position_a][fiabilite_a] = (amplitude, std_a)

    return amplitude, std_a

def calculate_gaussians_for_combinations(df):
    # dictionary to store 
    cache ={}   
    # unique values for FiabiliteA and PositionA
    unique_fiabilite_a = df['FiabiliteA'].unique()
    unique_position_a = df['PositionA'].unique()

    for position_a in unique_position_a:
        cache[position_a] = {}
        for fiabilite_a in unique_fiabilite_a:
            intensity_to_gaussian_param(df, cache, position_a, fiabilite_a)
            
    
    


#measure's a scenario's performance 
#computes imputs for auditory and visual systems and combines them
#runs the DNF model for nb_runs = 50
def _measure_scenario(X, DNF_params, I_audio, logpolar, visio_pos, visio_std, visio_amp = VISIO_AMP, nb_runs = NB_RUNS):
    #we do a Gaussian distribution of visual input
    I_visio = gaussian(X, visio_pos / 20., visio_std / 20., visio_amp, norm = False)
    if logpolar:
        L = reverse_log_matrix(X)
        I_visio = np.tensordot(L, I_visio, axes = ([2, 3], [0, 1])) if type(X) in [list, tuple] else np.matmul(L, I_visio)
    I = I_visio + I_audio
    outputs = []
    no_acts = False
    bombs = False
    expos = False
    for i in range(nb_runs + 1):
        test = DNF(X, I, **DNF_params)
        if i == 0: ### 1 longer run to check convergence
            test.run(deeper = True)
            if test.expo:
                expos = True
        else:
            test.run(deeper = False)
            if test.no_act:
                no_acts = True
            if test.bombing:
                bombs = True
            output_position = test.get_output()
            if logpolar:
                output_position = reverse_log_position(output_position)
            outputs.append(output_position)
    return np.mean(outputs), np.std(outputs), no_acts, bombs, expos

def evaluate_ab(DNF_params, audio_params, logpolar, dimension, scenarii = _SCENARII, x_min = X_MIN, x_max = X_MAX, dx = DX):
    if dimension == 1:
        X = np.arange(x_min, x_max + dx, dx)
    else:
        ### NOT MAINTAINED
        pass
    scen_avg = []
    scen_std = []
    scen_noa = []
    scen_bmb = []
    scen_exp = []
    for sce_params in scenarii:
        audio_pos = log_position(sce_params[0]/20.) if logpolar else sce_params[0]/20.
        I_audio = gaussian(X, audio_pos, 7.14972/20., 0.0573489, norm = False)
        visio_pos = sce_params[1]
        visio_std = sce_params[2]
        mod_avg, mod_std, no_act, bomb, expo = _measure_scenario(X, DNF_params, I_audio, logpolar, visio_pos, visio_std)
        scen_avg.append(mod_avg)
        scen_std.append(mod_std)
        scen_noa.append(no_act)
        scen_bmb.append(bomb)
        scen_exp.append(expo)
    return scen_avg, scen_std, scen_noa, scen_bmb, scen_exp
