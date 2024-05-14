#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np

from .utils import gaussian
from .preprocessing import *
from .model import DNF

VISIO_AMP = 1.
NB_RUNS = 50
X_MIN = -1.
X_MAX = 1.
DX = .02

# XP from Alais&Burr (2004)
#list of possible scenarios: [audio_pos, vis_pos, vis_std, expected_pos, expected_std]
_SCENARII = [
    [5., -5., 2., -4.074309897233062, -0.25936569201582255],
    [5., -5., 16., -0.20508982784627777, -0.06926217802519467],
    [5., -5., 32., 5.441596496251432, -0.06426838092295914],
    [2.5, -2.5, 2., -2.037767369486842, -0.19099566697287537],
    [2.5, -2.5, 16., 0.9191386013062262, -0.0778454260077265],
    [2.5, -2.5, 32., 2.463610182126206, -0.058144826422666177],
    [0., 0., 2., 0.3464517049164242, -0.38387585020875403],
    [0., 0., 16., 0.17004406818454337, -0.06119435961826628],
    [0., 0., 32., 0.6950725065369258, -0.10491158558924167],
    [-2.5, 2.5, 2., 2.50583325711363, -0.3005219709478384],
    [-2.5, 2.5, 16., 2.124202812555701, -0.06765820289937671],
    [-2.5, 2.5, 32., -0.050017480190087116, -0.06873127145127352],
    [-5., 5., 2., 5.057564920261513, -0.8727533045422469],
    [-5., 5., 16., 0.31183126692127494, -0.07459211375694377],
    [-5., 5., 32., -2.767600723603689, -0.06938425462302024]
]


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
    print("Scenarii is ", scenarii)
    for sce_params in scenarii:
        audio_pos = log_position(sce_params[0]/20.) if logpolar else sce_params[0]/20.
        I_audio = gaussian(X, audio_pos, audio_params['std']/20., audio_params['amp'], norm = False)
        visio_pos = sce_params[1]
        visio_std = sce_params[2]
        mod_avg, mod_std, no_act, bomb, expo = _measure_scenario(X, DNF_params, I_audio, logpolar, visio_pos, visio_std)
        scen_avg.append(mod_avg)
        scen_std.append(mod_std)
        scen_noa.append(no_act)
        scen_bmb.append(bomb)
        scen_exp.append(expo)
    return scen_avg, scen_std, scen_noa, scen_bmb, scen_exp
