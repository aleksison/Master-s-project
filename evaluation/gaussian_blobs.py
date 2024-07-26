import numpy as np
import matplotlib.pyplot as plt
from preprocessing import log_position

def gaussian(X, mean, std, coef=1., norm=False):
    if isinstance(X,(list, tuple)):
        X_h, X_v = X #for horizontal and vertical positions
        # Can be done with np.meshgrid
        X_h_grid = np.tile(X_h, (X_v.shape[0], 1))
        X_v_grid = np.tile(X_v, (X_h.shape[0], 1)).transpose()
        if isinstance(mean, (list, tuple)):
            mean_h, mean_v = mean
        else:
            mean_h = mean
            mean_v = 0.
        D2 = np.square(X_h_grid - mean_v) + np.square(X_v_grid - mean_h)
    else:
        D2 = np.square(X - mean)
    factor = coef / (np.sqrt(2.*np.pi) * std) if norm else coef
    return factor * np.exp(- D2 / (2. * np.square(std)))

X = np.arange(-1., 1. + .02, .02)
audio_pos = log_position(4/20.)
std = 26.0/20.
dB = [25,31,37,43]
amp = 0.018*dB[0]

print(gaussian(X,audio_pos, std, coef = amp, norm = False))

