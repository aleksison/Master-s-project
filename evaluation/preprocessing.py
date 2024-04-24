#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .utils import gaussian



def _1d_reverse_log_matrix(X, dx, A = 3., Bx = 1.4):
    L = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        u = A * (np.exp(np.abs(x) / Bx) - 1.) * np.sign(x)
        u /= np.pi
        if 0 <= u < dx:
            u_inf = np.argwhere(np.isclose(X, 0.))
            u_sup = u_inf + 1
        elif -dx < u < 0:
            u_sup = np.argwhere(np.isclose(X, 0.))
            u_inf = u_sup - 1
        elif u > X.max() or u < X.min():
            continue
        elif x > 0:
            u_inf = np.argmax(X * (X < u))
            u_sup = u_inf + 1
        else:
            u_sup = np.argmin(X * (X > u))
            u_inf = u_sup - 1
        L[i, u_inf] = (X[u_sup]-u)/(X[u_sup]-X[u_inf])
        L[i, u_sup] = (u-X[u_inf])/(X[u_sup]-X[u_inf])
    return L

def _2d_reverse_log_matrix(Xh, Xv, dx, A = 3., Bx = 1.4, By = 1.8):
    L = np.zeros((Xh.shape[0], Xv.shape[0], Xh.shape[0], Xv.shape[0])) #4D matrix
    for i, xh in enumerate(Xh):
        for j, xv in enumerate(Xv):
            if not np.isclose(xh, 0.) and np.abs(xh) < Bx * np.log(np.sqrt(1. + np.square(np.tan(xv / By)))):
                continue
            u = A * (np.exp(np.abs(xh) / Bx) / np.sqrt(1. + np.square(np.tan(xv / By))) - 1.) * np.sign(xh)
            u /= np.pi
            v = A * np.exp(np.abs(xh) / Bx) * np.tan(xv / By) / np.sqrt(1. + np.square(np.tan(xv / By)))
            v /= np.pi
            if 0 <= u <= dx:
                u_inf = np.argwhere(np.isclose(Xh, 0.))
                u_sup = u_inf + 1
            elif -dx <= u < 0:
                u_sup = np.argwhere(np.isclose(Xh, 0.))
                u_inf = u_sup - 1
            elif xh > 0:
                u_inf = np.argmax(Xh * (Xh < u))
                u_sup = u_inf + 1
            else:
                u_sup = np.argmin(Xh * (Xh > u))
                u_inf = u_sup - 1
            if 0 <= v < dx:
                v_inf = np.argwhere(np.isclose(Xv, 0.))
                v_sup = v_inf + 1
            elif -dx < v < 0:
                v_sup = np.argwhere(np.isclose(Xv, 0.))
                v_inf = v_sup - 1
            elif v > Xv.max() or v < Xv.min():
                continue
            elif xv > 0:
                v_inf = np.argmax(Xv * (Xv < v))
                v_sup = v_inf + 1
            else:
                v_sup = np.argmin(Xv * (Xv > v))
                v_inf = v_sup - 1

            L[i, j, u_inf, v_inf] += (Xh[u_sup]-u)/(Xh[u_sup]-Xh[u_inf]) * (Xv[v_sup]-v)/(Xv[v_sup]-Xv[v_inf])
            L[i, j, u_inf, v_sup] += (Xh[u_sup]-u)/(Xh[u_sup]-Xh[u_inf]) * (v-Xv[v_inf])/(Xv[v_sup]-Xv[v_inf])
            L[i, j, u_sup, v_inf] += (u-Xh[u_inf])/(Xh[u_sup]-Xh[u_inf]) * (Xv[v_sup]-v)/(Xv[v_sup]-Xv[v_inf])
            L[i, j, u_sup, v_sup] += (u-Xh[u_inf])/(Xh[u_sup]-Xh[u_inf]) * (v-Xv[v_inf])/(Xv[v_sup]-Xv[v_inf])
    return L

# https://stackoverflow.com/a/4104188
def run_once(f):
    def wrapper(*args, **kwargs):
        global L
        if not wrapper.has_run:
            wrapper.has_run = True
            f(*args, **kwargs)
        return L
    wrapper.has_run = False
    return wrapper

@run_once #these functions will be executed ony once even if called multiple times
def reverse_log_matrix(X, A = 3., Bx = 1.4, By = 1.8):
    global L
    if type(X) in [list, tuple]:
        Xh, Xv = X
        dx = Xh[1] - Xh[0]
        L = _2d_reverse_log_matrix(Xh, Xv, dx, A, Bx, By)
    else:
        dx = X[1] - X[0]
        L = _1d_reverse_log_matrix(X, dx, A, Bx)

def log_position(rho, A = 3., Bx = 1.4):
    rho_scale = np.abs(rho) * np.pi
    x = Bx * np.log(1 + rho_scale / A) * np.sign(rho) #applying the sign of the position
    return x

def reverse_log_position(x, A = 3., Bx = 1.4):
    rho = A * (np.exp(np.abs(x) / Bx) - 1.) * np.sign(x)
    rho /= np.pi
    return rho #does the inverse as to log_position

