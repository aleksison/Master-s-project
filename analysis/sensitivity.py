#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
from datetime import date
from SALib.sample import saltelli
from SALib.analyze import sobol

from evaluation import evaluate_ab

SAMPLE_SIZE = 5000
PATH = "/bettik/sforest/dabsa/sa/"



# From https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



### OBSOLETE
class SensitivityAnalysis:

    def __init__(self, param_names, param_lower, param_higher, model_name, sample_size = SAMPLE_SIZE):
        self.param_names = param_names[:8]
        self.name = model_name
        self.logpolar = param_lower[8]
        self.dim = param_lower[9]

        self.problem = {
            'num_vars': 8,
            'names': self.param_names,
            'bounds': list(zip(param_lower[:8], param_higher[:8]))
        }
        self.sample = saltelli.sample(self.problem, sample_size)

    def evaluate_ab_wrapper(self, new_params):
        DNF_params = {}
        DNF_params['tau'] = new_params[0]
        DNF_params['excit_amp'] = new_params[1]
        DNF_params['excit_std'] = new_params[2]
        DNF_params['inhib_amp'] = new_params[3]
        DNF_params['inhib_std'] = new_params[4]
        DNF_params['noise_amp'] = new_params[5]
        audio_params = {}
        audio_params['amp'] = new_params[6]
        audio_params['std'] = new_params[7]
        loss_mid, loss_slp, _ = evaluate_ab(DNF_params, audio_params, self.logpolar, self.dim)
        return loss_mid, loss_slp

    def evaluate_sample(self, save_step = None, path = PATH):
        self.outputs = np.zeros([self.sample.shape[0], 2])
        for i, params in enumerate(self.sample[42500:]): # TEMP
            self.outputs[i, 0], self.outputs[i, 1] = self.evaluate_ab_wrapper(params)

            if save_step is not None and i % save_step == 0:
                temp_results = {
                    "sample": self.sample[42500:42500+i], # TEMP
                    "outputs": self.outputs[:i, :]
                }
                with open(path + self.name + "_sample_2.json", 'w') as f: # TEMP
                    json.dump(temp_results, f, indent = 4, cls = NumpyEncoder)

    def run_analysis(self, path = PATH):
        Si_mid = sobol.analyze(self.problem, self.outputs[:, 0])
        Si_slp = sobol.analyze(self.problem, self.outputs[:, 1])
        results = {
            "analysis_mid": Si_mid,
            "analysis_slp": Si_slp,
            "param_names": self.param_names,
            "sample": self.sample,
            "outputs": self.outputs
        }
        with open(path + self.name + "_" + str(date.today()) + ".json", 'w') as f:
            json.dump(results, f, indent = 4, cls = NumpyEncoder)
