#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelmax

from .utils import *

DT = .01 #time step
KERNEL_SIZE = 4
MAX_CONV_TIME = 500
BOMB_THRESHOLD = 1.



class DNF:

    def __init__(self, X, I, tau, excit_amp, excit_std, inhib_amp, inhib_std, noise_amp, activation = "ReLU", kernel_size = KERNEL_SIZE):
        if type(X) in [list, tuple]: ### WARNING: 2D DNF not maintained for a while
            self.dim = 2
            self.dx = X[0][1] - X[0][0]
            self.field, _ = np.meshgrid(*X)
        else:
            self.dim = 1
            self.dx = X[1] - X[0]
            self.field = X#[:, np.newaxis]

        self.tau = tau
        self.e_a = excit_amp
        self.e_s = excit_std
        self.i_a = inhib_amp
        self.i_s = inhib_std
        self.n_a = noise_amp
        self.f = sigmoid if activation == "sigmoid" else ReLU

        self.e_s_norm = self.e_s / self.dx
        self.e_a_norm = self.e_a * np.sqrt(2.*np.pi) * self.e_s_norm
        self.i_s_norm = self.i_s / self.dx
        self.i_a_norm = self.i_a * np.sqrt(2.*np.pi) * self.i_s_norm

        self.no_act = False
        self.bombing = False
        self.expo = False

        self.U = np.zeros(self.field.shape) #initialize potential to zero
        self.noise = np.zeros(self.field.shape)
        self.I = I#[:, np.newaxis] if self.dim == 1 else I
        # kernel is a 1D array representing the kernel for the convolution operation
        # creates a symmetric kernel around 0
        kernel = np.arange(-kernel_size/2., kernel_size/2.+self.dx, self.dx)
        self.kernel = gaussian(kernel, 0, self.e_s, self.e_a) - gaussian(kernel, 0, self.i_s, self.i_a)

    ### OBSOLETE
    def generate_kernel(self):
        if self.kernel is None: 
            self.kernel = np.zeros((self.field.shape[0], self.field.shape[0]))
            for i in range(self.field.shape[0]):
                # ravel converts a multi-dimensional array into a one-dimensional
                self.kernel[i, :] = gaussian(self.field.ravel(), self.field[i], self.e_s, self.e_a) - gaussian(self.field.ravel(), self.field[i], self.i_s, self.i_a)
        return self.kernel

    def step(self, dt = DT): #computes lateral activation, applies noise, 
                            #and updates the field potential
        if self.dim == 1:
            #linear convolution between two one dimensional arrays
            lateral_activation = np.convolve(self.f(self.U), self.kernel, mode = 'valid')
            ### OBSOLETE
            # kernel = self.generate_kernel()
            # lateral_activation = np.matmul(kernel, self.f(self.U))
        else: ### Possibly outdated
            lateral_activation = self.e_a_norm * gaussian_filter(self.f(self.U), self.e_s_norm) \
                               - self.i_a_norm * gaussian_filter(self.f(self.U), self.i_s_norm)
        self.noise = noise(self.field.shape)
        self.U += dt / self.tau \
               * ( - self.U
                   + self.I
                   + lateral_activation
                   + self.n_a * self.noise
                 )

    def run(self, max_conv_time = MAX_CONV_TIME, bomb_thr = BOMB_THRESHOLD, deeper = False):
        ### Try once for a longer time to check stability
        max_time = 10 * max_conv_time if deeper else max_conv_time
        for t in range(max_time):
            self.step()
            if max_time - t == 100:
                U_mem_1 = self.U.max()
            elif max_time - t == 75:
                U_mem_2 = self.U.max()
            elif max_time - t == 50:
                U_mem_3 = self.U.max()
            elif max_time - t == 25:
                U_mem_4 = self.U.max()
        ### U < I when h = 0 means f(U) = 0 so no activity peak
        if self.U.max() < self.I.max():
            self.no_act = True
        ### "Bomb" means no point in running again, we're way too far from viable parameters
        if self.U.mean() > BOMB_THRESHOLD:
            self.bombing = True
        ### Check stability
        if np.isnan(self.get_output()) or U_mem_4 > U_mem_3 > U_mem_2 > U_mem_1:
            self.expo = True

    ### Weighted sum of f(U)
    def get_output(self): # here we do the weighted average of the field's potential
        if self.U.max() < 0: # AND IF IT'S EQUAL TO ZERO?
            return 0.
        elif self.dim == 2:
            return np.average(self.field[0, :], weights = self.f(self.U).max(axis = 1))
        return np.average(self.field, weights = self.f(self.U))

    ### Not very robust way of check for double selection
    def has_two_peaks(self):
        ### WARNING: "20" is arbitrary and depends on the size of X (number of neigbouring elements to consider)
        return (len(argrelmax(self.f(self.U), order = 20)[0]) > 1)
