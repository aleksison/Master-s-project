#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import json

from evaluation.utils import gaussian
from evaluation.preprocessing import log_position
from plots.activity import activity_evolution, activity_aggregation

pylab.rcParams.update({
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'})

X_MIN = -1.
X_MAX = 1.
DX = .02

MODEL_NAME = "1Dlin"
_SCENARII = [ ### ONLY USED FOR THE INPUTS
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

sce_params = _SCENARII[10] ### PICK SCENARIO NUMBER

with open("./config/config_" + MODEL_NAME + ".json") as f:
    params = json.load(f)
audio_std = params['values'][7]
audio_amp = params['values'][6]
logpolar = params['values'][8]

X = np.arange(X_MIN, X_MAX + DX, DX)
audio_pos = log_position(sce_params[0]/20.) if logpolar else sce_params[0]/20.
I_audio = gaussian(X, audio_pos, audio_std/20., audio_amp, norm = False)
visio_pos = sce_params[1]
visio_std = sce_params[2]

ax, I = activity_evolution(I_audio, visio_pos, visio_std, **dict(zip(params['names'], params['values'])))
activity_aggregation(ax, I, visio_pos, visio_std, **dict(zip(params['names'], params['values'])))
plt.tight_layout()
plt.show()
