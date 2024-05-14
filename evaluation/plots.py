import matplotlib.pyplot as plt
import numpy as np

from help import scenarios, mean7, std7

experimental_positions = [scenario[2] for scenario in scenarios]
experimental_stds = [scenario[3] for scenario in scenarios]

model_positions = mean7
model_stds = std7

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(experimental_positions, model_positions, color='#88c999', label='Model Positions')

# Add diagonal line for reference
plt.plot(experimental_positions, experimental_positions, color='red', linestyle='--', label='Ideal Agreement')

# Set plot labels and title
plt.xlabel('Actual Positions Guessed by Participants')
plt.ylabel('Model Positions')
plt.title('Comparison of Model Positions to Human Judgments')
plt.legend()

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate RMSE(root mean square error) for positions
rmse_positions = np.sqrt(np.mean((np.array(experimental_positions)-np.array(model_positions))**2))
print("RMSE for mean positions: ", rmse_positions)

# Calculate RMSE for standard deviations
rmse_stds = np.sqrt(np.mean((np.array(experimental_stds)-np.array(model_stds))**2))
print("RMSE for standard deviatons: ", rmse_stds)