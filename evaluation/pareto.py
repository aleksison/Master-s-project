import matplotlib.pyplot as plt
import numpy as np

v1 = [0.228, 0.952, 1.676, 2.4]
v2 = [0.075, 0.17223333, 0.26946667, 0.3667]

solutions1 = [
    (1.6742834742518526, 1.2390549626878253),
    (1.2846313495872521, 2.095360570618996),
    (1.4393947667328678, 2.6749238873291343),
    (1.562815538559787, 3.062125709080246)
]
solutions2 = [
    (1.0597202393184206, 1.5423668784927393),
    (1.2861130873922664, 1.9829273726340896),
    (1.5782291213649362, 2.3578194225523995),
    (1.748661516844479, 2.6628400123970244)
]
solutions1 = np.array(solutions1)
solutions2 = np.array(solutions2)

def pareto_front(solutions):
    is_pareto = np.ones(solutions.shape[0], dtype = bool)
    for i, c in enumerate(solutions):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(solutions[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return solutions[is_pareto]

pareto_points1 = pareto_front(solutions1)
pareto_points2 = pareto_front(solutions2)

plt.figure(figsize=(10,6))

plt.scatter(solutions1[:,0], solutions1[:,1], color = 'pink', label='All Solutions')
#Annotation of visual stds for each solution
for i, (x,y) in enumerate(solutions1):
    plt.annotate(str(v1[i]),(x,y), textcoords="offset points", xytext=(0,10), ha='center')

plt.scatter(pareto_points1[:,0], pareto_points1[:,1], color='red', label='Pareto Front')
plt.plot(pareto_points1[:,0], pareto_points1[:,1], color='red', linestyle='--')

plt.scatter(solutions2[:,0], solutions2[:,1], color = 'lightblue', label='All Solutions')
#Annotation of visual stds for each solution
for i, (x,y) in enumerate(solutions2):
    plt.annotate(str(v2[i]),(x,y), textcoords="offset points", xytext=(0,10), ha='center')

plt.scatter(pareto_points2[:,0], pareto_points2[:,1], color='blue', label='Pareto Front')
plt.plot(pareto_points2[:,0], pareto_points2[:,1], color='blue', linestyle='--')


plt.xlabel('RMSE for Mean Positions')
plt.ylabel('RMSE for Standard Deviations')
plt.title('Pareto Front for Given Solutions')
plt.legend()
plt.grid(True)
plt.show()