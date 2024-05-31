import matplotlib.pyplot as plt
import numpy as np

vis_std = [6, 8, 10, 11, 12, 13.8, 16.6, 20]
solutions = [
    (1.4203352207966566, 2.891484828614372),
    (1.3368525552909512, 2.40523605302266),
    (1.3305642038390029, 2.022840210172587),
    (1.3465762691319318, 1.8660671764754928),
    (1.4064348353663318, 1.6956443682292983),
    (1.5069867155084975, 1.4903321875602566),
    (1.6742834742518526, 1.2390549626878253),
    (1.9520009172826704, 1.0636422693587961)
]
solutions = np.array(solutions)

def pareto_front(solutions):
    is_pareto = np.ones(solutions.shape[0], dtype = bool)
    for i, c in enumerate(solutions):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(solutions[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return solutions[is_pareto]

pareto_points = pareto_front(solutions)

plt.figure(figsize=(10,6))
plt.scatter(solutions[:,0], solutions[:,1], color = 'pink', label='All Solutions')

#Annotation of visual stds for each solution
for i, (x,y) in enumerate(solutions):
    plt.annotate(str(vis_std[i]),(x,y), textcoords="offset points", xytext=(0,10), ha='center')

plt.scatter(pareto_points[:,0], pareto_points[:,1], color='red', label='Pareto Front')
plt.plot(pareto_points[:,0], pareto_points[:,1], color='red', linestyle='--')

plt.xlabel('RMSE for Mean Positions')
plt.ylabel('RMSE for Standard Deviations')
plt.title('Pareto Front for Given Solutions')
plt.legend()
plt.grid(True)
plt.show()