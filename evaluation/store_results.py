import pandas as pd

with open('scenarios.txt', 'r') as file:
    lines = file.readlines()

arrays = []

for line in lines:
    array_str = line.strip()[1:-1] # to remove the brackets
    array = [float(item.strip()) for item in array_str.split(',')] #convert string elements to float
    arrays.append(array)

df = pd.DataFrame(arrays)

df.to_excel('results.xlsx', index=False, header=False)