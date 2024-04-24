#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import json
from datetime import date

from evaluation.alais_burr import evaluate_ab

D_MULT = 20
PATH = "/bettik/forestsi/redac_beta/"



class Grid:

    def __init__(self, init_param_names, init_param_values, param_min, param_max, set_name):
        self.p_names = init_param_names
        self.p_values = init_param_values
        self.p_min = param_min
        self.p_max = param_max
        self.name = set_name

    def evaluate_ab_wrapper(self, **kwargs):
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

    def evaluate_with_interaction(self, idx1, idx2, repeat = 1, save_step = None, d_mult = D_MULT, path = PATH):
        pmin1, pmax1 = self.p_min[idx1], self.p_max[idx1]
        pmin2, pmax2 = self.p_min[idx2], self.p_max[idx2]
        param1_range = np.arange(pmin1, pmax1 + (pmax1-pmin1)/d_mult, (pmax1-pmin1)/d_mult)
        param2_range = np.arange(pmin2, pmax2 + (pmax2-pmin2)/d_mult, (pmax2-pmin2)/d_mult)

        Z_avg = np.zeros((param1_range.shape[0], param2_range.shape[0], 15))
        Z_std = np.zeros((param1_range.shape[0], param2_range.shape[0], 15))
        Z_no_act = np.zeros((param1_range.shape[0], param2_range.shape[0], 15))
        Z_bomb = np.zeros((param1_range.shape[0], param2_range.shape[0], 15))
        Z_expo = np.zeros((param1_range.shape[0], param2_range.shape[0], 15))
        params = copy.deepcopy(self.p_values)
        for i, x in enumerate(param1_range):
            params[idx1] = x
            for j, y in enumerate(param2_range):
                params[idx2] = y
                Z_avg[i, j, :], Z_std[i, j, :], Z_no_act[i, j, :], Z_bomb[i, j, :], Z_expo[i, j, :] = self.evaluate_ab_wrapper(**dict(zip(self.p_names, params)))

        return {
            "all_params": dict(zip(self.p_names, self.p_values)),
            "x_label": self.p_names[idx1],
            "y_label": self.p_names[idx2],
            "x_ticks": param1_range.tolist(),
            "y_ticks": param2_range.tolist(),
            "Z_avg": Z_avg.tolist(),
            "Z_std": Z_std.tolist(),
            "Z_no_act": Z_no_act.tolist(),
            "Z_bomb": Z_bomb.tolist(),
            "Z_expo": Z_expo.tolist()
        }


    def evaluate_pairs(self, pairs, repeat = 1, save_step = None, path = PATH):
        for p in pairs:
            results = self.evaluate_with_interaction(*p, repeat = repeat, save_step = save_step, path = path)
            with open(path + self.name + "_" + str(date.today()) + "_" + results['x_label'] + "-" + results['y_label'] + ".json", 'w') as f:
                json.dump(results, f, indent = 4)



class ParetoGrid(Grid):

    def __init__(self, init_param_names, init_param_values, param_min, param_max, set_name, ea_list, es_list):
        super(ParetoGrid, self).__init__(init_param_names, init_param_values, param_min, param_max, set_name)
        self.ea_list = ea_list
        self.es_list = es_list

    def evaluate_ab_wrapper(self, **kwargs):
        DNF_params = {}
        DNF_params['tau'] = kwargs['tau']
        DNF_params['excit_amp'] = self.ea_list[round(int(kwargs['t12']))]
        DNF_params['excit_std'] = self.es_list[round(int(kwargs['t12']))] / 20.
        DNF_params['inhib_amp'] = kwargs['inhib_amp']
        DNF_params['inhib_std'] = kwargs['inhib_std']
        DNF_params['noise_amp'] = kwargs['noise_amp']
        audio_params = {}
        audio_params['amp'] = kwargs['audio_amp']
        audio_params['std'] = kwargs['audio_std']
        logpolar = kwargs['logpolar']
        dimension = kwargs['dimension']
        return evaluate_ab(DNF_params, audio_params, logpolar, dimension)
