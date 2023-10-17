import numpy as np
import pandas as pd
import pickle




def smooth(dsm,smooth_side = 2):
    '''
    Smooths the DSM to avoid contour lines effects by replacing the altitude of each point with the average altitude of a smoothing  box.

    Inputs:
     - dsm (np.ndarray): 2D numpy matrix representing the digital surface model with elevation values for each point.
     - smooth_side (int): Length of the side of the smoothing kernel.

    Outputs:
     - dsm_smoothed (np.ndarray): Smoothed digital surface model (DSM).
    '''
    
    dsm_smoothed = dsm.copy()
    
    for idx_row in range(smooth_side,len(dsm_smoothed)-smooth_side):
        for idx_col in range(smooth_side,len(dsm_smoothed[0])-smooth_side):
            
            dsm_smoothed[idx_row,idx_col] = np.mean(dsm[idx_row-smooth_side:idx_row+smooth_side,
                                                    idx_col-smooth_side:idx_col+smooth_side])
            
    return dsm_smoothed




def gradient(z, resolution = 2):
    '''
    Computes the gradients of the digital surface model (DSM) from north to south and west to east.

    Inputs:
     - z (np.ndarray): 2D numpy matrix representing the digital surface model with elevation values for each point.
     - resolution (float): The resolution scale of the DSM.

    Outputs:
     - Tuple[np.ndarray, np.ndarray]: Tuple containing the altitude gradients from west to east (p) and from north to south (q).
    '''
    
    p = np.zeros(z.shape)
    q = np.zeros(z.shape)
    
    for row in range(len(z)):
        for col in range(len(z[0])):
            
            if row == 0 or row == len(z)-1:
                continue
            
            if col == 0 or col == len(z[0])-1:
                continue
            
            zN = z[row-1][col] #righe vanno da nord a sud!!
            zS = z[row+1][col]
            zE = z[row][col+1] #colonne vanno da ovest a est
            zW = z[row][col-1]
            
            p[row][col] = (zE-zW)/(2*resolution) #grad da west a est
            q[row][col] = (zS-zN)/(2*resolution) #grad da nord a sud
    
    #padding row
    p[0,:] = p[1,:]
    p[len(p)-1,:]=p[len(p)-2,:]
    
    q[0,:] = q[1,:]
    q[len(p)-1,:]=q[len(q)-2,:]
    
    #padding col
    p[:,0] = p[:,1]
    p[:,len(p[0])-1]=p[:,len(p[0])-2]
    
    q[:,0] = q[:,1]
    q[:,len(q[0])-1]=q[:,len(q[0])-2]
    
    return p,q
    


 
          

def lapserate(tc,h,p):
    
    '''
    Computes the lapse rate correction to multiply to the height for variations in temperature with height (adiabatic lapse rate).

    Inputs:
     - tc (float): Reference temperature in degrees Celsius.
     - h (float): Specific humidity.
     - p (float): Atmospheric pressure in Pascal.

    Outputs:
     - lr (float): Lapse rate correction to be multiplied to the height.
    '''
    e0 = 610.8 * np.exp(17.27 * tc / (tc + 237.3))
    ws = 0.622 * e0 / p
    rh = (h / ws) * 100
    rh = np.where(rh < 100, rh, 100)
    ea = e0 * (rh / 100)
    rv = 0.622 * ea / (p - ea)
    lr = 9.8076 * (1 + (2501000 * rv) / (287 * (tc + 273.15))) / (1003.5 +
            (0.622 * 2501000 ** 2 * rv) / (287 * (tc + 273.15) ** 2))
    lr = lr * -1
    
    
    return lr  



def compute_specific_humidity(tdew, tc, p):
    '''
    Computes the specific humidity.
    Inputs:
     - tdew (float): The dewpoint temperature (°C).
     - tc (float): The reference temperature (°C).
     - p (float): The atmospheric pressure (Pa).
    Output:
     - hs (float): The specific humidity.
    '''

    ea = 610.8 * np.exp(17.27 * tdew / (tdew + 237.3))
    hs = 0.622 * ea/(p-0.378*ea)
    return hs



def compute_relative_humidity(tdew, tc):
    '''
    Computes the relative humidity.
    Inputs:
     - tdew (float): The dewpoint temperature (°C).
     - tc (float): The reference temperature (°C).
    Output:
     - hr (float): The relative humidity (%).
    '''
    
    ea = 0.6108 * np.exp(17.27 * tdew / (tdew + 237.3))
    e0 = 0.6108 * np.exp(17.27 * tc/(tc + 237.3))
    hr = (ea/e0) * 100
    return hr



def compute_specific_from_relative_humidity(hr, p, tc):
    '''
    Computes the specific humidity from the relative humidity.
    Inputs:
     - hr (float): The relative humidity (%).
     - p (float): The atmospheric pressure (Pa).
     - tc (float): The reference temperature (°C).
    Output:
     - hs (float): The specific humidity.
    '''
    
    e0 = 610.8 * np.exp(17.27 * tc/(tc + 237.3))
    ea = hr/100*e0
    hs = 0.622 * ea/(p-0.378*ea)
    return hs
    
    

    
    
    