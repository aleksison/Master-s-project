import matplotlib.pyplot as plt
import numpy as np

# Data from the table
lateral_excitation = [0.1, 0.325, 0.55, 0.775, 1.0]
lateral_inhibition = [0.05, 0.0875, 0.125, 0.1625, 0.2]

# RMSE values for each pair (excitation, inhibition)
rmse_mean_positions = [
    [4.2163, 4.1111, None, None, None],
    [1.2723, 1.6606, 2.5565, 2.9606, 3.1827],
    [1.5109, 1.4033, 1.2986, 1.3134, 1.2780],
    [1.8925, 1.7346, 1.6682, 1.5830, 1.5387],
    [2.1478, 2.0211, 1.9858, 1.8683, 1.7951]
]

rmse_std_dev = [
    [4.1510, 3.9261, None, None, None],
    [2.0060, 2.7438, 3.2943, 3.4531, 3.3990],
    [1.2961, 1.3046, 1.2632, 1.2468, 1.2643],
    [1.3866, 1.4956, 1.5400, 1.5822, 1.4716],
    [1.4910, 1.6962, 1.8090, 1.8586, 1.8817]
]

# Plot RMSE of mean positions
plt.figure(figsize=(12, 6))
for i, exc in enumerate(lateral_excitation):
    rmse_vals = rmse_mean_positions[i]
    rmse_vals = [v for v in rmse_vals if v is not None]  # Remove None values
    inhib_vals = lateral_inhibition[:len(rmse_vals)]
    plt.plot(inhib_vals, rmse_vals, label=f'Excitation = {exc}')

plt.title('RMSE of Mean Positions vs. Lateral Inhibition Amplitude')
plt.xlabel('Lateral Inhibition Amplitude')
plt.ylabel('RMSE of Mean Positions')
plt.legend()
plt.grid(True)
plt.show()

# Plot RMSE of standard deviations
plt.figure(figsize=(12, 6))
for i, exc in enumerate(lateral_excitation):
    rmse_vals = rmse_std_dev[i]
    rmse_vals = [v for v in rmse_vals if v is not None]  # Remove None values
    inhib_vals = lateral_inhibition[:len(rmse_vals)]
    plt.plot(inhib_vals, rmse_vals, label=f'Excitation = {exc}')

plt.title('RMSE of Standard Deviations vs. Lateral Inhibition Amplitude')
plt.xlabel('Lateral Inhibition Amplitude')
plt.ylabel('RMSE of Standard Deviations')
plt.legend()
plt.grid(True)
plt.show()
