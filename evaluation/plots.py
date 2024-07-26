import matplotlib.pyplot as plt
import numpy as np

from sansbruit import scenarios, exc15

experimental_positions = [scenario[2] for scenario in scenarios]
#experimental_stds = [scenario[3] for scenario in scenarios]

model_positions = exc15
#model_stds = std21

# Different colours for congruent and not congruent scenarios
#colours = ['#88c999' if scenario[6] else 'blue' for scenario in scenarios_points]

# Scatter plot
#plt.figure(figsize=(8,6))
#plt.scatter(experimental_positions, model_positions, color=colours, label='Model Positions')

# Define colors for congruent and incongruent stimuli
congruent_color = '#88c999'
incongruent_color = 'blue'

# Scatter plot for congruent stimuli
plt.scatter([pos for i, pos in enumerate(experimental_positions) if scenarios[i][6]], 
            [pos for i, pos in enumerate(model_positions) if scenarios[i][6]], 
            color=congruent_color, label='Congruent stimuli')

# Scatter plot for incongruent stimuli
plt.scatter([pos for i, pos in enumerate(experimental_positions) if not scenarios[i][6]], 
            [pos for i, pos in enumerate(model_positions) if not scenarios[i][6]], 
            color=incongruent_color, label='Incongruent stimuli')


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
#rmse_stds = np.sqrt(np.mean((np.array(experimental_stds)-np.array(model_stds))**2))
#print("RMSE for standard deviations: ", rmse_stds)