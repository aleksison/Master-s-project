#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from itertools import product
from datetime import date
import json

from evaluation import evaluate_ab

DELTA = .01
LEARNING_RATE = .01
MAX_STEPS = 200
PATH = "/bettik/sforest/dabsa/gd/"



### OBSOLETE
class GradientDescent:

    def __init__(self, init_param_names, init_param_values, set_name, path = PATH):
        self.param_names = init_param_names[:8]
        self.param_values = np.array(init_param_values[:8])
        self.logpolar = init_param_values[8]
        self.dim = init_param_values[9]
        self.filename = path + set_name + "_" + str(date.today()) + ".json"
        self.step_nb = 0

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

    def step(self, delta = DELTA, learning_rate = LEARNING_RATE):
        coefs = product([1. - delta/2., 1. + delta/2.], repeat = 8)
        inputs = []
        for c in coefs:
            inputs.append([self.param_values[i]*c[i] for i in range(len(c))])
        outputs_mid = np.zeros(2**8)
        outputs_slp = np.zeros(2**8)
        norm_outputs = np.zeros(2**8)
        for i, params in enumerate(inputs):
            loss_mid, loss_slp = self.evaluate_ab_wrapper(params)
            outputs_mid[i] = loss_mid
            outputs_slp[i] = loss_slp
        norm_outputs_mid = (outputs_mid - outputs_mid.mean()) / outputs_mid.std()
        norm_outputs_slp = (outputs_slp - outputs_slp.mean()) / outputs_slp.std()
        norm_outputs = np.sqrt((norm_outputs_mid**2+norm_outputs_slp**2) / 2.)
        norm_outputs = norm_outputs.reshape((2, 2, 2, 2, 2, 2, 2, 2))
        grad = [gr[0, 0, 0, 0, 0, 0, 0, 0] for gr in np.gradient(norm_outputs)]
        results = {
            "step_nb": self.step_nb,
            "params": dict(zip(self.param_names, self.param_values)),
            "best_loss_mid": outputs_mid.min(),
            "best_loss_slp": outputs_slp.min(),
            "best_loss_norm": norm_outputs.min(),
            "gradient": grad
        }
        self.param_values -= learning_rate * np.array(grad)
        self.step_nb += 1
        return results

    def run(self, max_steps = MAX_STEPS):
        # TODO: check for convergence
        for _ in range(max_steps):
            results = self.step()
            with open(self.filename, 'a') as f:
                json.dump(results, f, indent = 4)
