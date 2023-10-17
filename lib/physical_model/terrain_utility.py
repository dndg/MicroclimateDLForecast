import numpy as np
import pandas as pd
from utility import gradient

'''
This python module contains methods to compute the morphological properties of the terrain.
'''


def terrain(dsm, attrib):    
    '''
    Computes the slope and aspect of the digital surface model (DSM).
    
    Inputs:
     - dsm (np.ndarray): 2D numpy matrix representing the digital surface model with elevation values for each point.
     - attrib (str): Parameter to decide whether to compute slope or aspect. Valid values are 'slope' and 'aspect'.

    Outputs:
     - np.ndarray: If attrib is 'slope', returns the slope of the DSM as a 2D numpy matrix.
                    If attrib is 'aspect', returns the aspect of the DSM as a 2D numpy matrix.
    '''
    
    if attrib == 'aspect':
        p,q = gradient(dsm)
        q = q*(-1) #the rows goes from north to south but our sdr goes towards north
        aspect = -90*(1-np.sign(q))*(1-np.abs(np.sign(p)))+180*(1+np.sign(p))-180/np.pi*np.sign(p)*np.arccos(-q/np.sqrt(q**2+p**2+10e-10))
        return aspect
    
    elif attrib == 'slope':
        x,y = gradient(dsm)
        slope = (np.arctan(np.sqrt(x*x + y*y))) * 180/np.pi
        return slope
    
    else:
        print('unknown attribute')
        
        



def horizonangle(dsm,azimuth,reso = 1):
    '''
    Computes the tangent of the horizon angle for each cell in a specified azimuth.

    Inputs:
     - dsm (np.ndarray): 2D numpy matrix representing the digital surface model with elevation values for each point.
     - azimuth (float): Azimuthal angle in degrees that defines the portion of the horizon to compute the angle.
                        0 corresponds to North, 90 corresponds to East, 180 corresponds to South, and 270 corresponds to West.
     - reso (float): The resolution scale of the DSM.

    Outputs:
     - horizon (np.ndarray): 2D numpy array containing the tangent of the horizon angle for each cell in the specified azimuth.
    '''
    
    dsm = np.nan_to_num(dsm) 
    azi = (azimuth)*np.pi/180

    padx = dsm.shape[1]
    pady = dsm.shape[0]

    horizon = np.zeros(dsm.shape)
    shape_dsm3 = (dsm.shape[0]+2*pady,dsm.shape[1]+2*padx)
    dsm3 = np.zeros(shape_dsm3) #padded dsm
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
                       
        
    return horizon



def mean_horizonangle(dsm, steps = 36, reso = 1):
    
    '''
    Computes the mean horizon angle for each point of the DSM grid by dividing the horizon into sectors.

    Parameters:
     - dsm (np.ndarray): 2D numpy matrix representing the digital surface model with elevation values for each point.
     - steps (int): Number of sectors to divide the horizon.
     - reso (float): The resolution scale of the DSM.

    Returns:
     - ha (np.ndarray): 2D numpy array containing the mean horizon angle in decimal degrees for each cell,
                        averaged over all the specified sectors.
    '''
    
    dsm = np.nan_to_num(dsm)
    ha = np.zeros(dsm.shape)

    for s in range(1,steps+1,1):
        horizon_angle = np.arctan(horizonangle(dsm, s * 360 / steps, reso))* (180 / np.pi)
        ha = ha + horizon_angle
    
    ha = ha / steps
    
    return ha




def skyviewfactor(dsm, steps = 36, reso = 100):
    
    '''
    Computes the mean horizon angle and sky view factor for each point of the DSM grid.

    Parameters:
     - dsm (np.ndarray): 2D numpy matrix representing the digital surface model with elevation values for each point.
     - steps (int): Number of sectors to divide the horizon.
     - reso (float): The resolution scale of the DSM.

    Returns:
     - Tuple[np.ndarray, np.ndarray]: A tuple containing two 2D numpy arrays. 
            - The first array contains the mean horizon angle in decimal degrees for each cell, averaged over all the specified sectors.
            - The second array contains the sky view factor for each cell.
    '''
    
    ha = mean_horizonangle(dsm, steps, reso)
    svf = 0.5 * np.cos(2 * ha * (np.pi / 180)) + 0.5
    
    return ha, svf
