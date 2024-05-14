#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .preprocessing import *
from .model import DNF
from .utils import *

VISIO_STD = 16.6 # this is an average of 2, 16, 32
NB_RUNS = 50
X_MIN = -1.
X_MAX = 1.
DX = .02

# XP from Alais&Burr (2004)

df = pd.read_excel('dataVentriloquie.xlsx')
cond_dict = avg_std_par_condition(df)
#print(std_dict)
#print(cond_dict)
#print(len(cond_dict)) # 512 correct
filtered_cond_dict = {key: value for key, value in cond_dict.items() if not np.isnan(value).any()}
#print(filtered_cond_dict)
#print(len(filtered_cond_dict))
#list of possible scenarios: [audio_pos, vis_pos, expected_pos, expected_std, audio_rel, visio_rel]
_SCENARII = []
for key in filtered_cond_dict:
    audio_pos = key[2]
    vis_pos = key[3]
    expected_pos, expected_std = filtered_cond_dict[key]
    audio_rel = key[0]
    vis_rel = key[1]
    _SCENARII.append([audio_pos, vis_pos, expected_pos, expected_std, audio_rel, vis_rel])

print(_SCENARII)
print("Number of scenarios: ", len(_SCENARII))
   
#measure's a scenario's performance 
#computes imputs for auditory and visual systems and combines them
#runs the DNF model for nb_runs = 50
def _measure_scenario(X, DNF_params, I_audio, logpolar, visio_pos, visio_amp, visio_std = VISIO_STD, nb_runs = NB_RUNS):
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

def evaluate_ab(DNF_params, audio_params, visio_params, logpolar, dimension, scenarii = _SCENARII, x_min = X_MIN, x_max = X_MAX, dx = DX):
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
        # We take the parameter a and multiple it by intensity for each case
        audio_rel = sce_params[4]
        audio_pos = log_position(sce_params[0]/20.) if logpolar else sce_params[0]/20.
        amp_key = audio_params['a'] # and what do we do for intensities with gabor? 0.5 etc
        if audio_rel == 25:
            amp_key = audio_params['a']*25
        elif audio_rel == 31:
            amp_key = audio_params['a']*31
        elif audio_rel == 37:
            amp_key = audio_params['a']*37
        elif audio_rel == 43:
            amp_key = audio_params['a']*43
        I_audio = gaussian(X, audio_pos, audio_params['std']/20., amp_key, norm = False)
        visio_pos = sce_params[1]
        visio_rel = sce_params[5]
        visio_v = visio_params['v']
        visio_amp = visio_v
        if visio_rel == 3:
           visio_amp = visio_v*3
        elif visio_rel == 6:
           visio_amp = visio_v*6
        elif visio_rel == 9:
            visio_amp = visio_v*9
        elif visio_rel == 12:
            visio_amp = visio_v*12
        mod_avg, mod_std, no_act, bomb, expo = _measure_scenario(X, DNF_params, I_audio, logpolar, visio_pos, visio_amp)
        scen_avg.append(mod_avg)
        scen_std.append(mod_std)
        scen_noa.append(no_act)
        scen_bmb.append(bomb)
        scen_exp.append(expo)
    return scen_avg, scen_std, scen_noa, scen_bmb, scen_exp
