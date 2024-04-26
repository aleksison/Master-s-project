#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from evaluation.utils import gaussian
from evaluation.preprocessing import log_position, reverse_log_matrix
from evaluation.model import DNF



VISIO_AMP = 1.
X_MIN = -1.
X_MAX = 1.
DX = .02
MAX_TIME = 500



def model_output():
    pass

def activity_evolution(I_audio, visio_pos, visio_std, visio_amp = VISIO_AMP, max_time = MAX_TIME, x_min = X_MIN, x_max = X_MAX, dx = DX, **kwargs):
    DNF_params = {}
    DNF_params['tau'] = kwargs['tau']
    DNF_params['excit_amp'] = kwargs['excit_amp']
    DNF_params['excit_std'] = kwargs['excit_std']
    DNF_params['inhib_amp'] = kwargs['inhib_amp']
    DNF_params['inhib_std'] = kwargs['inhib_std']
    DNF_params['noise_amp'] = kwargs['noise_amp']
    logpolar = kwargs['logpolar']
    dimension = kwargs['dimension']

    X = np.arange(x_min, x_max + dx, dx)
    I_hist = np.zeros((X.shape[0], max_time))
    U_hist = np.zeros((X.shape[0], max_time))
    I_max = np.zeros((max_time))
    U_bary = np.zeros((max_time))

    I_visio = gaussian(X, visio_pos / 20., visio_std / 20., visio_amp, norm = False)
    if logpolar:
        L = reverse_log_matrix(X)
        I_visio = np.tensordot(L, I_visio, axes = ([2, 3], [0, 1])) if type(X) in [list, tuple] else np.matmul(L, I_visio)
    I = I_visio + I_audio
    print(20.*X[np.argmax(I)])
    print(20.*np.average(X, weights = I))

    model = DNF(X, I, **DNF_params)
    for t in range(max_time):
        model.step()
        U_hist[:, t] = model.U
        # I_hist[:, t] = I + model.noise
        # U_hist[:, t] = model.f(model.U)
        I_max[t] = 20.*X[np.argmax(I + model.noise)]
        U_bary[t] = 20.*np.average(X, weights = model.f(model.U))

    # model = DNF(X, I, **DNF_params)
    # model.run()
    # print(model.expo)
    # print(model.has_two_peaks())

    Y = np.arange(max_time)
    X, Y = np.meshgrid(X, Y) if dimension == 1 else np.meshgrid(X[0], Y)
    # f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    f, a0 = plt.subplots()

    # div0 = make_axes_locatable(a0)
    # cax0 = div0.append_axes('right', size='5%', pad=0.05)
    # div1 = make_axes_locatable(a1)
    # cax1 = div1.append_axes('right', size='5%', pad=0.05)

    # pcm0 = a0.pcolormesh(20.*X.transpose(), Y.transpose(), I_hist, cmap = cm.gist_yarg, shading = 'auto')
    # a0.plot(U_bary, range(max_time), c = 'white')
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(20.*X.transpose(), Y.transpose(), U_hist, cmap = cm.gist_yarg, linewidth = 0, antialiased = True)
    # plt.plot(20.*X, I)
    # plt.plot(range(max_time), 20.*:X[0, np.argmax(U_hist, axis = 0)])
    # f.colorbar(pcm0, cax=cax0, orientation='vertical')
    a0.set_xlabel('$x$')
    a0.set_ylabel("Time")
    # plt.tight_layout()
    # plt.subplot(212)
    # pcm1 = a1.pcolormesh(20.*X.transpose(), Y.transpose(), I_hist, cmap = cm.gist_yarg, shading = 'auto')
    # # a0.plot(20. * np.arange(x_min, x_max + dx, dx), I_visio, label = '$I_V$')
    # # a0.plot(20. * np.arange(x_min, x_max + dx, dx), I_audio, label = '$I_A$')
    # # a0.plot(20. * np.arange(x_min, x_max + dx, dx), I, label = '$I$')
    # # plt.legend()
    # # plt.plot(I_max, range(max_time))
    # f.colorbar(pcm1, cax=cax1, orientation='vertical')
    # a1.set_xlabel('$x$')
    # a1.set_ylabel("Time")
    # a1.get_yaxis().set_ticks([])
    return a0, I

def activity_aggregation(ax, I, visio_pos, visio_std, visio_amp = VISIO_AMP, num_tests = 30, max_time = MAX_TIME, x_min = X_MIN, x_max = X_MAX, dx = DX, **kwargs):
    DNF_params = {}
    DNF_params['tau'] = kwargs['tau']
    DNF_params['excit_amp'] = kwargs['excit_amp']
    DNF_params['excit_std'] = kwargs['excit_std']
    DNF_params['inhib_amp'] = kwargs['inhib_amp']
    DNF_params['inhib_std'] = kwargs['inhib_std']
    DNF_params['noise_amp'] = kwargs['noise_amp']
    logpolar = kwargs['logpolar']
    dimension = kwargs['dimension']

    X = np.arange(x_min, x_max + dx, dx)
    U_hist = np.zeros((num_tests, max_time))

    for i in range(num_tests):
        model = DNF(X, I, **DNF_params)
        for t in range(max_time):
            model.step()
            U_hist[i, t] = 20.*np.average(X, weights = model.f(model.U))

    Y = np.arange(max_time)
    X, Y = np.meshgrid(X, Y) if dimension == 1 else np.meshgrid(X[0], Y)
    for i in range(num_tests):
        ax.plot(U_hist[i, :], range(max_time))

    U_end_mean = np.mean(U_hist[:, -1])
    U_end_sd = np.std(U_hist[:, -1])
    ax.plot(20.*np.arange(x_min, x_max + dx, dx), 500-gaussian(20.*np.arange(x_min, x_max + dx, dx), U_end_mean, U_end_sd, 100.), c = "black")
