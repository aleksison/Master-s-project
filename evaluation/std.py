import numpy as np
import pandas as pd
from scipy.stats import norm

df = pd.read_excel('dataVentriloquie.xlsx')



# las desviaciones se guardan en un diccionario
std_dict = {}    

fiabilites_v = df[df['FiabiliteV'] != 0]['FiabiliteV'].unique()
positions_v = df[df['PositionV'] != 0]['PositionV'].unique()

for pos in positions_v:
    for fiab in fiabilites_v:
        filtered_df = df[(df['PositionA'] == 0) & (df['FiabiliteV'] == fiab) & (df['PositionV'] == pos)]
        estimated_positions = np.array(filtered_df['X'])
        std = np.std(estimated_positions)
        std_dict[(fiab,pos)] = std

