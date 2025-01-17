#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json

from evaluation.alais_burr_new import evaluate_ab
#from sklearn.model_selection import GridSearchCV

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
else:
    MODEL_NAME = "1Dlog"



def evaluate_ab_wrapper(dnf_exc, a, v, **kwargs):
	DNF_params = {}
	DNF_params['tau'] = kwargs['tau']
	DNF_params['excit_amp'] = dnf_exc[0] #kwargs['excit_amp']
	DNF_params['excit_std'] = dnf_exc[1] #kwargs['excit_std']
	DNF_params['inhib_amp'] = kwargs['inhib_amp']
	DNF_params['inhib_std'] = kwargs['inhib_std']
	DNF_params['noise_amp'] = 0 #kwargs['noise_amp']
	audio_params = {}
	audio_params['a'] = a
	audio_params['std'] = kwargs['audio_std']
	visio_params = {}
	visio_params['v'] = v
	#visio_params['std'] = kwargs['visio_std']  #could be extracted to the json file later on
	logpolar = kwargs['logpolar']
	dimension = kwargs['dimension']
	return evaluate_ab(DNF_params, audio_params, visio_params, logpolar, dimension)



with open("./config/config_" + MODEL_NAME + ".json") as f:
    params = json.load(f)

#Finetunning parameters for audio and visual stimuli
# [0.075   0.17223333 0.26946667 0.3667    ] [0.004 0.018 0.032 0.046] v2, a2
v1 = np.linspace(0.228, 2.4, 4) # gabor [0.228 0.952 1.676 2.4]
v2 = np.linspace(0.075, 0.3667, 4) #so that the amplitude is around 1, points
#Finetunnning DNF parameters 
exc_amp = np.linspace(0.1, 1., num = 5)
inh_amp = np.linspace(0.05, 0.2, num = 5)
exc_std = np.linspace(0.2, 2., num = 5)
inh_std = np.linspace(2., 100., num = 5)

print("Widths of lateral excitation: ", exc_std)
print("Widths of lateral inhibition: " , inh_std)
'''
ab_combinations = [(exc, inh) for exc in exc_std for inh in inh_std]
valid_combinations = [(A,B) for A, B in ab_combinations if A<B ]
print("Widths' combinations: " , valid_combinations)
dnf_stds = valid_combinations[11]
'''
# Exploring lateral excitation, tuples of amplitude and width
# amplitude range (0.1, 1) - selected 0.425
# width range (0.2, 2) - selected 0.85
exc_combinations = [(amp, wid) for amp in exc_amp for wid in exc_std]
valid_exc_comb = [(A, B) for A, B in exc_combinations if A > 0.15]
print('Excitation combinations: ', valid_exc_comb)
dnf_exc = valid_exc_comb[19]


a2 = np.linspace(0.004,0.046, 4) #for any value of dB -> amp between 0.1 and 2

a_val = a2[1] # the optimal value 0.018
v_val = v2[3] # it' either V2 for points or V1 for gabor


NB_META = 1 # para cada scenario 25 ejecuciones
NB_SCEN = 128 # theoretically it is 512, on practice it is 256
meta_avg = np.zeros((NB_SCEN, NB_META))
meta_std = np.zeros((NB_SCEN, NB_META))
for i in range(NB_META):
	meta_avg[:, i], meta_std[:, i], _, _, _ = evaluate_ab_wrapper(dnf_exc, a_val, v_val, **dict(zip(params['names'], params['values'])))
mod_avgs = 20. * np.median(meta_avg, axis = 1)
mod_stds = 20. * np.median(meta_std, axis = 1)

print(np.array2string(mod_avgs, separator=', '))
print(np.array2string(mod_stds, separator=', '))