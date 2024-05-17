#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np



def noise(shape):
    return np.random.randn(*shape)

def sigmoid(X):
    return 1. / (1. + np.exp(-X))

def ReLU(X):
    return np.maximum(X, 0.)

def gaussian(X, mean, std, coef = 1., norm = False):
    if type(X) in [list, tuple]:
        X_h, X_v = X #for horizontal and vertical positions
        # Can be done with np.meshgrid
        X_h_grid = np.tile(X_h, (X_v.shape[0], 1))
        X_v_grid = np.tile(X_v, (X_h.shape[0], 1)).transpose()
        if type(mean) in [list, tuple]:
            mean_h, mean_v = mean
        else:
            mean_h = mean
            mean_v = 0.
        D2 = np.square(X_h_grid - mean_v) + np.square(X_v_grid - mean_h)
    else:
        D2 = np.square(X - mean)
    factor = coef / (np.sqrt(2.*np.pi) * std) if norm else coef
    return factor * np.exp(- D2 / (2. * np.square(std)))

def calculate_std(df):
    std_dict = {}    

    fiabilites_v = df[df['FiabiliteV'] != 0]['FiabiliteV'].unique()
    positions_v = df[df['PositionV'] != 0]['PositionV'].unique()

    for pos in positions_v:
      for fiab in fiabilites_v:
         filtered_df = df[(df['PositionA'] == 0) & (df['FiabiliteV'] == fiab) & (df['PositionV'] == pos)]
         estimated_positions = np.array(filtered_df['X'])
         std_dict[(pos, fiab)] = estimated_positions.std()
    return std_dict

def avg_std_par_condition(df):
    condition_dict = {}
    # unique values of reliabilities and positions
    unique_fiabilite_a = df[df['FiabiliteA'] != 0]['FiabiliteA'].unique()
    unique_fiabilite_v = df[df['FiabiliteV'] != 0]['FiabiliteV'].unique()
    unique_position_a = df[df['PositionA'] != 0]['PositionA'].unique()
    unique_position_v = df[df['PositionV'] != 0]['PositionV'].unique()
    for fiab_a in unique_fiabilite_a:
        for fiab_v in unique_fiabilite_v:
            for pos_a in unique_position_a:
                for pos_v in unique_position_v:
                # data frame for a specific combination of the 4 parameters
                    filtered_df = df[(df['FiabiliteA'] == fiab_a) & (df['FiabiliteV'] == fiab_v) &
                                 (df['PositionA'] == pos_a) & (df['PositionV'] == pos_v) & (df['stimV'] == "points")]
                    avg_x = np.mean(filtered_df['X'])
                    std_x = np.std(filtered_df['X'])
                    
                    condition_dict[(fiab_a, fiab_v, pos_a, pos_v)] = (avg_x, std_x)
    return condition_dict