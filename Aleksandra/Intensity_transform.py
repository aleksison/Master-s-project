import warnings
import numpy as np
import pandas as pd

from scipy.stats import norm

#cache = {}

df = pd.read_excel('dataVentriloquie.xlsx')
# function that calculates standard deviation for a certain intensity
def intensity_to_gaussian_param(df, cache, position_a, fiabilite_a):
    # Hay que usar dynamic programming
    if position_a in cache and fiabilite_a in cache[position_a]:
        return cache[position_a][fiabilite_a]
    # Las entradas de solo audio 
    filtered_df = df[(df['PositionV'] == 0) & (df['FiabiliteA'] == fiabilite_a) & (df['PositionA'] == position_a)]
    # desviacion de las posiciones estimadas
    std_a = filtered_df['X'].std()
    amplitude = 1 / (std_a * np.sqrt(2 * np.pi))

    if position_a not in cache:
        cache[position_a] = {}
    cache[position_a][fiabilite_a] = (amplitude, std_a)

    return amplitude, std_a

def calculate_gaussians_for_combinations(df):
    # dictionary to store 
    cache ={}   
    # unique values for FiabiliteA and PositionA
    unique_fiabilite_a = df.loc[df['FiabiliteA'] != 0, 'FiabiliteA'].unique()
    unique_position_a = df.loc[df['PositionA'] != 0, 'PositionA'].unique()

    for position_a in unique_position_a:
        #cache[position_a] = {}
        for fiabilite_a in unique_fiabilite_a:
            intensity_to_gaussian_param(df, cache, position_a, fiabilite_a)
    return cache

cache = calculate_gaussians_for_combinations(df)
print(cache)

def mean_amp_std(cache):
    amplitudes = []
    std_devs = []
    for x in cache.values():
        for amplitude, std_dev in x.values():
            amplitudes.append(amplitude)
            std_devs.append(std_dev)
    mean_amp = np.mean(amplitudes)
    mean_std_dev = np.mean(std_devs)
    print("Mean Amplitude: ", mean_amp)
    print("Mean Standard Deviation: ", mean_std_dev)

mean_amp_std(cache)