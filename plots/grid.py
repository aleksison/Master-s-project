#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib import cm
import json

SAVE_DIR = "outputs/parameter_plot/"
NUM_POINTS = 100000

# _SCENARII = np.array([
#     [5., -5., 2., -4.074309897233062, -0.25936569201582255],
#     [5., -5., 16., -0.20508982784627777, -0.06926217802519467],
#     [5., -5., 32., 5.441596496251432, -0.06426838092295914],
#     [2.5, -2.5, 2., -2.037767369486842, -0.19099566697287537],
#     [2.5, -2.5, 16., 0.9191386013062262, -0.0778454260077265],
#     [2.5, -2.5, 32., 2.463610182126206, -0.058144826422666177],
#     [0., 0., 2., 0.3464517049164242, -0.38387585020875403],
#     [0., 0., 16., 0.17004406818454337, -0.06119435961826628],
#     [0., 0., 32., 0.6950725065369258, -0.10491158558924167],
#     [-2.5, 2.5, 2., 2.50583325711363, -0.3005219709478384],
#     [-2.5, 2.5, 16., 2.124202812555701, -0.06765820289937671],
#     [-2.5, 2.5, 32., -0.050017480190087116, -0.06873127145127352],
#     [-5., 5., 2., 5.057564920261513, -0.8727533045422469],
#     [-5., 5., 16., 0.31183126692127494, -0.07459211375694377],
#     [-5., 5., 32., -2.767600723603689, -0.06938425462302024]
# ])
#
# Exp_avg = np.array(_SCENARII[:, 3])
# Exp_std = -1 / np.array(_SCENARII[:, 4]) / np.sqrt(2*np.pi)

Exp_avg = np.array([-4.18, -0.33, 4.18, -2.16, 0.26, 1.37, -0.13, -0.59, -0.39, 2.16, 1.57, -0.98, 5.10, -0.13, -3.01])
Exp_std = np.array([1.57, 5.49, 5.95, 2.02, 6.14, 6.34, 0.98, 5.94, 4.51, 1.37, 4.51, 5.88, 1.57, 5.49, 5.75])



# From https://stackoverflow.com/a/40239615
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def _convert_label(lbl):
    return {
        "tau": r'$\tau$',
        "excit_amp": r'$\lambda_+$',
        "excit_std": r'$\sigma_+$',
        "inhib_amp": r'$\lambda_-$',
        "inhib_std": r'$\sigma_-$',
        "audio_amp": r'$\lambda_A$',
        "audio_std": r'$\sigma_A$',
        "noise_amp": r'$\sigma_N$',
        "t12": r'$p_+$'
    }[lbl]

def grayscale(filename, save_file = None, save_dir = SAVE_DIR, pareto = False):
    with open(filename) as f:
        results = json.load(f)
    X, Y = np.array(results['x_ticks']), np.array(results['y_ticks'])
    X_RED = results['all_params'][results['x_label']]
    Y_RED = results['all_params'][results['y_label']]
    if results['x_label'] in ["excit_std", "inhib_std"]:
        X *= 20
        X_RED *= 20
    if results['y_label'] in ["excit_std", "inhib_std"]:
        Y *= 20
        Y_RED *= 20
    X, Y = np.meshgrid(X, Y)

    Z_bomb = np.mean(results['Z_bomb'], axis = 2)
    Zm0 = np.ma.masked_less(Z_bomb, .5)
    Z_expo = np.mean(results['Z_expo'], axis = 2)
    Zm1 = np.ma.masked_less(Z_expo, .5)
    Z_no_act = np.mean(results['Z_no_act'], axis = 2)
    Zm2 = np.ma.masked_less(Z_no_act, .5)

    Z_avg = 20. * np.array(results['Z_avg'])
    Z_loss_avg = np.sqrt(np.mean(np.square(Z_avg - Exp_avg), axis = 2))
    Z_std = 20. * np.array(results['Z_std'])
    Z_loss_std = np.sqrt(np.mean(np.square(Z_std - Exp_std), axis = 2))
    # Z_loss_both = np.sqrt((np.square(Z_loss_avg) + np.square(Z_loss_std)) / 2)
    Z_loss_both = np.log(Z_loss_avg * Z_loss_std)

    # TEST
    Z_loss_avg[Z_bomb == True] = np.nan
    Z_loss_avg[Z_expo == True] = np.nan
    Z_loss_avg[Z_no_act == True] = np.nan
    Z_loss_avg = np.nan_to_num(Z_loss_avg, nan = 2.5)
    # Z_loss_avg = gaussian_filter(Z_loss_avg, sigma = 1)

    Z_loss_std[Z_bomb == True] = np.nan
    Z_loss_std[Z_expo == True] = np.nan
    Z_loss_std[Z_no_act == True] = np.nan
    Z_loss_std = np.nan_to_num(Z_loss_std, nan = 4)
    # Z_loss_std = gaussian_filter(Z_loss_std, sigma = 1)

    Z_loss_both[Z_bomb == True] = np.nan
    Z_loss_both[Z_expo == True] = np.nan
    Z_loss_both[Z_no_act == True] = np.nan
    Z_loss_both = np.nan_to_num(Z_loss_both, nan = 3)
    # Z_loss_both = gaussian_filter(Z_loss_both, sigma = 1)

    # PARETO
    if pareto:
        outputs_mid = np.array(Z_loss_avg).flatten()
        outputs_slp = np.array(Z_loss_std).flatten()
        sample_X = X.transpose().flatten()
        sample_Y = Y.transpose().flatten()
        outputs = np.concatenate((outputs_mid[:, np.newaxis], outputs_slp[:, np.newaxis]), axis = 1)
        # outputs = outputs[~np.isnan(outputs).any(axis=1)]
        # outputs = outputs[~np.isclose(outputs, 0).all(axis=1)]
        pareto_mask = is_pareto_efficient(outputs)
        pareto_outputs = outputs[pareto_mask, :]
        pareto_X = sample_X[pareto_mask]
        pareto_Y = sample_Y[pareto_mask]
        # print(pareto_X, pareto_Y, pareto_outputs)
        pareto_sort = pareto_outputs[:, 0].argsort()
        pareto_X = pareto_X[pareto_sort]
        pareto_Y = pareto_Y[pareto_sort]

    plt.figure()
    # plt.pcolormesh(X.transpose(), Y.transpose(), Z_loss_avg, cmap = cm.gist_gray, shading = 'auto', vmin = 0.5, vmax = 2.5)
    levels = np.linspace(0.75, 2.5, 8)
    plt.contourf(X.transpose(), Y.transpose(), Z_loss_avg, cmap = cm.gist_gray, levels = levels, extend = 'both')
    plt.colorbar()
    if results['x_label'] != "t12":
        plt.scatter(X_RED, Y_RED, marker = '+', c = 'r')
    plt.title("Loss AVG")
    plt.xlabel(_convert_label(results['x_label']))
    plt.ylabel(_convert_label(results['y_label']))
    plt.pcolor(X.transpose(), Y.transpose(), Zm0, hatch='X', alpha=0., shading = 'auto')
    plt.pcolor(X.transpose(), Y.transpose(), Zm1, hatch='/', alpha=0., shading = 'auto')
    plt.pcolor(X.transpose(), Y.transpose(), Zm2, hatch='.', alpha=0., shading = 'auto')
    if pareto:
        plt.scatter(pareto_X, pareto_Y, marker = 'x')
    if results['x_label'] == "excit_amp" and results['y_label'] == "excit_std":
        pX = [0.27, 0.28, 0.425, 0.55, 0.9]
        pY = [2.0, 1.5, 0.85, 0.5, 0.25]
        tck, u = splprep([pX, pY])
        xi, yi = splev(np.linspace(0, 1, NUM_POINTS), tck)
        plt.plot(xi, yi)
    if save_file is not None:
        plt.savefig(save_dir + "mid_" + save_file) # TEMP

    plt.figure()
    # plt.pcolormesh(X.transpose(), Y.transpose(), Z_loss_std, cmap = cm.gist_gray, shading = 'auto', vmin = 1, vmax = 4)
    levels = np.linspace(1, 4, 7)
    plt.contourf(X.transpose(), Y.transpose(), Z_loss_std, cmap = cm.gist_gray, levels = levels, extend = 'both')
    plt.colorbar()
    if results['x_label'] != "t12":
        plt.scatter(X_RED, Y_RED, marker = '+', c = 'r')
    plt.title("Loss STD")
    plt.xlabel(_convert_label(results['x_label']))
    plt.ylabel(_convert_label(results['y_label']))
    plt.pcolor(X.transpose(), Y.transpose(), Zm0, hatch='X', alpha=0., shading = 'auto')
    plt.pcolor(X.transpose(), Y.transpose(), Zm1, hatch='/', alpha=0., shading = 'auto')
    plt.pcolor(X.transpose(), Y.transpose(), Zm2, hatch='.', alpha=0., shading = 'auto')
    if pareto:
        plt.scatter(pareto_X, pareto_Y, marker = 'x')
    if results['x_label'] == "excit_amp" and results['y_label'] == "excit_std":
        pX = [0.27, 0.28, 0.425, 0.55, 0.9]
        pY = [2.0, 1.5, 0.85, 0.5, 0.25]
        tck, u = splprep([pX, pY])
        xi, yi = splev(np.linspace(0, 1, NUM_POINTS), tck)
        plt.plot(xi, yi)
    if save_file is not None:
        plt.savefig(save_dir + "slp_" + save_file) # TEMP

    # plt.figure()
    # # plt.pcolormesh(X.transpose(), Y.transpose(), Z_loss_both, cmap = cm.gist_gray, shading = 'auto', vmin = 0, vmax = 3)
    # levels = np.linspace(0, 3, 7)
    # plt.contourf(X.transpose(), Y.transpose(), Z_loss_both, cmap = cm.gist_gray, levels = levels, extend = 'max')
    # plt.colorbar()
    # if results['x_label'] != "t12":
    #     plt.scatter(X_RED, Y_RED, marker = '+', c = 'r')
    # plt.title("Loss Both")
    # plt.xlabel(_convert_label(results['x_label']))
    # plt.ylabel(_convert_label(results['y_label']))
    # plt.pcolor(X.transpose(), Y.transpose(), Zm0, hatch='X', alpha=0., shading = 'auto')
    # plt.pcolor(X.transpose(), Y.transpose(), Zm1, hatch='/', alpha=0., shading = 'auto')
    # plt.pcolor(X.transpose(), Y.transpose(), Zm2, hatch='.', alpha=0., shading = 'auto')
    # if pareto:
    #     plt.scatter(pareto_X, pareto_Y, marker = 'x')
    # if results['x_label'] == "excit_amp" and results['y_label'] == "excit_std":
    #     pX = [0.27, 0.28, 0.425, 0.55, 0.9]
    #     pY = [2.0, 1.5, 0.85, 0.5, 0.25]
    #     tck, u = splprep([pX, pY])
    #     xi, yi = splev(np.linspace(0, 1, NUM_POINTS), tck)
    #     plt.plot(xi, yi)
    # if save_file is not None:
    #     plt.savefig(save_dir + "both_" + save_file) # TEMP

    if save_file is None:
        plt.show()
    else:
        plt.close()
