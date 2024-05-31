import matplotlib.pyplot as plt

# Data 
visual_stds = [6, 8, 10, 11, 12, 13.8, 16.6, 20]
rmse_mean_positions = [1.4203352207966566, 1.3368525552909512, 1.3305642038390029, 1.3465762691319318, 
                       1.4064348353663318, 1.5069867155084975, 1.6742834742518526, 1.9520009172826704]
rmse_mean_deviations = [2.891484828614372, 2.40523605302266, 2.022840210172587, 1.8660671764754928,
                        1.6956443682292983, 1.4903321875602566,1.2390549626878253, 1.0636422693587961]

# Plotting
plt.figure(figsize=(10,6))
plt.plot(visual_stds, rmse_mean_positions, marker='o', label='RMSE for Mean Positions')
plt.plot(visual_stds, rmse_mean_deviations, marker='o', label='RMSE for Standard Deviations')
plt.xlabel('Visual Standard Deviation')
plt.ylabel('RMSE')
plt.title('RMSE vs Visual Standard Deviation')
plt.legend()
plt.grid(True)

plt.show()