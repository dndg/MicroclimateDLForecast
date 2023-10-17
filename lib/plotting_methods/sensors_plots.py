import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_sensors_data(df, len_rows, len_cols, variable,
                      figsize = (25,14), color = 'orange', x_fontsize = 25,y_fontsize = 25,title_fontsize = 30):
    '''
    Plots the data collected by the sensors.
    Inputs:
     - df (pd.DataFrame): The pandas DataFrame containing the sensors data.
     - len_rows (int): Length of the subplot rows.
     - len_cols (int): Length of the subplot columns.
     - variable (str): Name of the variable to plot. Can be 'Temperature' or 'Humidity'.
     - figsize (tuple): Dimension of the figure.
     - color (str, optional): Color of the plot. Defaults to 'orange'.
     - x_fontsize (int, optional): Fontsize of the x-axis label. Defaults to 25.
     - y_fontsize (int, optional): Fontsize of the y-axis label. Defaults to 25.
     - title_fontsize (int, optional): Fontsize of the title label. Defaults to 30.
    '''

    fig,axs = plt.subplots(len_rows, len_cols, figsize = figsize, sharex=False,sharey=True)

    idx_col = 0
    idx_row = 0
    
    for idx_sensor in df.Sensor_id.unique():
        dfs = df[df.Sensor_id == idx_sensor]
    
        t = np.arange(len(dfs))/24
    
        ax = axs[idx_row,idx_col]
        ax.plot(t, dfs[variable].values, color = color)
    
        ax.set_xlabel('Days',fontsize = x_fontsize)
        
        if idx_col == 0:
            if variable == 'Temperature':
                ax.set_ylabel('Temperature [°C]',fontsize = y_fontsize)
            if variable == 'Humidity':
                ax.set_ylabel('Humidity [%]',fontsize = y_fontsize)
        
        ax.set_title('Sensor {}'.format(idx_sensor), fontsize = title_fontsize)
        
        idx_col = idx_col +1
        if idx_col == len_cols:
            idx_col = 0
            idx_row = idx_row+1
        
    #axs[len_rows-1,5].axis('off')  
    fig.tight_layout()
    