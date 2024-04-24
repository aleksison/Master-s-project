import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

df = pd.read_excel('dataVentriloquie.xlsx')

position_v  = 4
fiabilite_v = 9

filtered_df = df[(df['PositionA'] == 0) & (df['FiabiliteV'] == fiabilite_v) & (df['PositionV'] == position_v)]
stimulus_positions = np.array(filtered_df['PositionV']) # hay que recogerlos del archivo en funcion de la intensidad 
estimated_positions = np.array(filtered_df['X'])


# Define Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

# Function to fit Gaussian to data
def fit_gaussian(stimulus_positions, estimated_positions):
    # Perform curve fitting
    popt, _ = curve_fit(gaussian, stimulus_positions, estimated_positions, p0=[1, np.mean(stimulus_positions), np.std(stimulus_positions)])
    
    # Extract parameters
    amplitude, mean, stddev = popt

    # Generate points for the fitted Gaussian curve
    x_fit = np.linspace(min(stimulus_positions), max(stimulus_positions), 100)
    y_fit = gaussian(x_fit, *popt)
    
    plt.scatter(stimulus_positions, estimated_positions, label='Data')
    plt.plot(x_fit,y_fit, 'r-', label='Fitted Gaussian')
    plt.xlabel('Stimulus Position')
    plt.ylabel('Estimated Position')
    plt.legend()
    plt.title('Fitted Gaussian to Stimulus Position Estimates')
    plt.show()

    print("Amplitude: ", amplitude)
    print("Mean: ", mean)
    print("Standard Deviation: ", stddev)

fit_gaussian(stimulus_positions, estimated_positions)
    


