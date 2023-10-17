import numpy as np
import pandas as pd



def windheight(ui, zi, zo,roughness_height = 0.023):
    
    '''
    Applies a logarithmic height correction to compute the wind speed at zo, knowing the wind speed at zi.

    Inputs:
     - ui (float): Wind speed measured at height zi.
     - zi (float): Height (m) at which ui has been measured.
     - zo (float): Height (m) at which to compute the wind speed.
     - roughness_height (float): Height of surface roughness.

    Output:
     - uo (float): Wind speed computed at height zo.
   
    '''
            
    uo = ui * np.log(zo/roughness_height) / np.log(zi/roughness_height)
    return uo  






def windcoef(dsm, direction, hgt = 1, reso = 1):
    
    '''
    Computes the topographic shelter coefficient for scaling the wind speed.

    Inputs:
     - dsm (np.ndarray): Digital surface model. It is a 2D numpy matrix with elevation in each point.
     - direction (float): Direction from which the wind is blowing in decimal degrees.
     - hgt (float): Height parameter.
     - reso (float): The dsm resolution scale.

    Outputs:
     - index (np.ndarray): 2D array containing the topographic shelter coefficient for each point.
    '''

    dsm = np.nan_to_num(dsm) 

    azi = direction * (np.pi / 180)

    padx = dsm.shape[1]
    pady = dsm.shape[0]

    horizon = np.zeros(dsm.shape)
    shape_dsm3 = (dsm.shape[0]+2*pady,dsm.shape[1]+2*padx)
    dsm3 = np.zeros(shape_dsm3) 
    y = dsm.shape[0]
    x = dsm.shape[1]
    dsm3[pady:(y + pady), padx:(x + padx)] = dsm


    for step in range(1,np.max([y,x]),1):

        scaled_x_start = int(padx+ np.sin(azi) * step)
        if scaled_x_start>padx+x or scaled_x_start<0: #if x_start esce dalla box paddata
            break
        scaled_x_stop = int(x+int(padx+ np.sin(azi) * step))

        scaled_y_start = int(pady+ np.cos(azi-np.pi) * step)
        if scaled_y_start>pady+y or scaled_y_start<0: #if y_start esce dalla box paddata
            break
        scaled_y_stop = int(y+int(pady+ np.cos(azi-np.pi) * step))

        search_grid = dsm3[scaled_y_start:scaled_y_stop, scaled_x_start:scaled_x_stop]
        ref_grid = dsm3[pady:(y + pady), padx:(x + padx)]

        horizon = np.maximum(horizon, (search_grid-ref_grid)/(step*reso))
        horizon = np.where(horizon >= (hgt/(step*reso)), horizon, 0)

    index = 1 - np.arctan(0.17 * 100 * horizon) / (np.pi/2) #100*horizon it is the slope percent
    
    return index




                      