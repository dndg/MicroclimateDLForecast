import numpy as np
import pandas as pd
import richdem as rd

from terrain_utility import terrain, horizonangle, mean_horizonangle, skyviewfactor
from solar_utility import julday, solartime, solalt, solazi




def solarindex(dsm, slope, aspect, localtime, lat, long, julian, reso):
    
    '''
    This method calculates the solar index for each point in the area of interest. The solar index is represented as a 2D matrix that is multiplied by the direct solar radiation to describe its local variation within the study area.

    Inputs:
     - dsm (np.ndarray): The digital surface model, represented as a 2D numpy matrix with elevation values for each point.
     - slope (np.ndarray): The slope of the DSM, represented as a 2D numpy matrix with slope values for each point.
     - aspect(np.ndarray): The aspect of the DSM, represented as a 2D numpy matrix with aspect values for each point.
     - localtime (float): The hour measured by a smartphone located in the area of interest.
     - lat (float): The latitude of the center of the DSM.
     - long (float): The longitude of the center of the DSM.
     - julian (float): The Julian day measured by a smartphone located in the area of interest.
     - reso (float): The resolution scale of the DSM.
     
    Outputs:
     - s_index (np.ndarray): The solar index for each point in the area of interest.
    '''
    
    saltitude = solalt(localtime, lat, long, julian)
    szenith = np.pi/2 - saltitude
    sazimuth = solazi(localtime, lat, long, julian)
    
    slope = slope * (np.pi / 180)
    aspect = aspect * (np.pi / 180)
    
    shadowmask = np.ones(slope.shape)
    
    horizon_angle = horizonangle(dsm, sazimuth*180/np.pi, reso)
    
    #mette zone ombra quando sole è coperto
    shadowmask[horizon_angle > np.tan(saltitude)] = 0
    
    s_index = np.zeros(slope.shape)
    s_index = np.cos(szenith) * np.cos(slope) + np.sin(szenith) * np.sin(slope) * np.cos(sazimuth - aspect)
        
    s_index[s_index < 0] = 0
    s_index =s_index * shadowmask/(np.cos(szenith)+10e-15)
    
    return s_index



def shortwave_radiation(dir, dif, julian, localtime, lat, long,
                        dsm, slope, aspect, ha, svf, reso,
                        alb = 0.25, albr = 0.25):
    
    '''
    This method calculates the shortwave radiation for each point in the study area by combining the direct and diffuse radiation. The incoming shortwave radiation is conventionally represented with a positive sign. This method does not consider canopy effects.

    Inputs:
     - dir (float): The direct radiation on a horizontal surface (MJ/m**2/hour).
     - dif (float): The diffuse radiation on a horizontal surface (MJ/m**2/hour).
     - julian (float): The Julian day measured by a smartphone located in the area of interest.
     - localtime (float): The hour measured by a smartphone located in the area of interest.
     - lat (float): The latitude of the center of the DSM.
     - long (float): The longitude of the center of the DSM.
     - dsm (np.ndarray): The digital surface model, represented as a 2D numpy matrix with elevation values for each point.
     - slope (np.ndarray): The slope of the DSM, represented as a 2D numpy matrix with slope values for each point.
     - aspect (np.ndarray): The aspect of the DSM, represented as a 2D numpy matrix with aspect values for each point.
     - ha (np.ndarray): The mean horizon angle.
     - svf (np.ndarray): The sky view factor.
     - reso (float): The resolution scale of the DSM.
     - alb (float): The albedo of the area of interest.
     - albr (float): The albedo of the area of interest, used for the reflected diffuse radiation.
     
    Outputs:  
     - swr (np.ndarray): The shortwave radiation for each point in the area of interest.
    '''
    
    
    s_index = solarindex(dsm,slope, aspect, localtime, lat, long, julian, reso)
    
    #direct radiation
    dirr = dir * s_index
    
    #diffuse radiation
    a = slope * (np.pi / 180)
    
    k = dir / 4.87       
    k = np.where(k < 1, k, 1)
    
    isor = 0.5 * dif * (1 + np.cos(a)) * (1 - k) * svf
    
    cisr = k * dif * s_index 
    sdi = (slope + ha) * (np.pi / 180)
    refr = 0.5 * albr * (1 - np.cos(sdi)) * dif
    
    difr = isor + cisr + refr
    
    sw2r = difr + dirr
    swr = (1 - alb) * sw2r
    
    return  swr



def longwave_radiation(h, tc, n, p, svf, dsm_shape = (187,214)):
    
    '''
    This method calculates the longwave radiation for each point in the study area by combining the black body radiation emitted from the surface and the black body radiation incoming from the atmosphere. The outgoing longwave radiation is conventionally represented with a positive sign. This method does not consider canopy effects.

    Inputs: 
     - h (float): The specific humidity.
     - tc (np.ndarray): The reference temperature (°C).
     - n (float): The fractional cloud cover.
     - p (float): The atmospheric pressure (Pa).
     - svf (np.ndarray): The sky view factor.
     - dtm_shape: The shape of the digital surface model. The DTM of the area of interest has a shape of (187, 214).
     
    Output:
     - lwr (np.ndarray): The longwave radiation for each point in the area of interest.
    
    '''
    
    if np.array(h).shape == ():
        h = np.full(dsm_shape, h)
        
    if np.array(tc).shape == ():
        tc = np.full(dsm_shape, tc)
        
    if np.array(p).shape == ():
        p = np.full(dsm_shape, p)
    
    if np.array(n).shape == ():
        n = np.full(dsm_shape, n)
    
    #compute vapor pressure 
    e0 = 610.8 * np.exp(17.27 * tc / (tc + 237.3))
    ws = 0.622 * e0 / p
    rh = (h / ws) * 100
    rh = np.where(rh < 100, rh, 100)
    e = e0 * (rh / 100)

    ecs = 1.24*(e/(tc + 273.15))**(1/7) #1.24 relationship between vapor pressure and temperature near the ground Brutsaert (1975).
    Fn = (1-n)+(1/ecs)*n
    
    em = ecs*Fn
    sigma = 2.043e-10 #Boltzman constant expressed in MJ/m**2/hour/K**4 (5.669*10-8*3600/10-6)
    
    lwr = (1 - em*svf) *sigma * (tc + 273.15)**4
    return lwr
    
    

    




