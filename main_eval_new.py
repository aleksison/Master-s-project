#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json

from evaluation.alais_burr_new import evaluate_ab

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
else:
    MODEL_NAME = "1Dlog"



def evaluate_ab_wrapper(**kwargs):
	DNF_params = {}
	DNF_params['tau'] = kwargs['tau']
	DNF_params['excit_amp'] = kwargs['excit_amp']
	DNF_params['excit_std'] = kwargs['excit_std']
	DNF_params['inhib_amp'] = kwargs['inhib_amp']
	DNF_params['inhib_std'] = kwargs['inhib_std']
	DNF_params['noise_amp'] = kwargs['noise_amp']
	audio_params = {}
	audio_params['amp'] = kwargs['audio_amp']
      

	audio_params['std'] = kwargs['audio_std']
	logpolar = kwargs['logpolar']
	dimension = kwargs['dimension']
	return evaluate_ab(DNF_params, audio_params, logpolar, dimension)



with open("./config/config_" + MODEL_NAME + ".json") as f:
    params = json.load(f)
   

NB_META = 25 # para cada scenario 25 ejecuciones
NB_SCEN = 256 # theoretically it is 512, on practice it is 256
meta_avg = np.zeros((NB_SCEN, NB_META))
meta_std = np.zeros((NB_SCEN, NB_META))
for i in range(NB_META):
	meta_avg[:, i], meta_std[:, i], _, _, _ = evaluate_ab_wrapper(**dict(zip(params['names'], params['values'])))
mod_avgs = 20. * np.median(meta_avg, axis = 1)
mod_stds = 20. * np.median(meta_std, axis = 1)

print(np.array2string(mod_avgs, separator=', '))
print(np.array2string(mod_stds, separator=', '))