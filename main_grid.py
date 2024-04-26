#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
from scipy.interpolate import splprep, splev

from analysis.grid import Grid, ParetoGrid ### WARNING: update PATH

NUM_POINTS = 10000

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
else:
    MODEL_NAME = "1Dlog"

### Pick pairs of parameters to test together
pairs = [(1, 2), (3, 4), (1, 3), (2, 4), (6, 7), (0, 5)]
# pairs = [(0, 1), (0, 2), (0, 3), (2, 3)]
# pairs = [(0, 4), (1, 4), (1, 5), (2, 5)]
# pairs = [(3, 5), (4, 5), (0, 6), (0, 7)]
# pairs = [(1, 6), (1, 7), (2, 6), (2, 7)]
# pairs = [(3, 6), (4, 6), (5, 6)]
# pairs = [(3, 7), (4, 7), (5, 7)]

if len(sys.argv) > 2:
    pairs = [pairs[int(sys.argv[2])]]



with open("./config/config_" + MODEL_NAME + ".json") as f:
    params = json.load(f)

### WARNING: update PATH in analysis/grid.py
if MODEL_NAME == "croissant": ### Replacing "excit_amp" and "excit_std" with "t12"
    pX = [0.27, 0.28, 0.38, 0.55, 0.9]
    pY = [2.0, 1.5, 0.8, 0.5, 0.25]
    tck, u = splprep([pX, pY])
    xi, yi = splev(np.linspace(0, 1, NUM_POINTS), tck)
    grid = ParetoGrid(params['names'], params['values'], params['min'], params['max'], MODEL_NAME, xi, yi)
    grid.evaluate_pairs([(1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)])
else:
    grid = Grid(params['names'], params['values'], params['min'], params['max'], MODEL_NAME)
    grid.evaluate_pairs(pairs)
