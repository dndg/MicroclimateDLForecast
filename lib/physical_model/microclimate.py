import numpy as np
import pandas as pd

from solar_utility import julday
from utility import lapserate
from radiation_utility import  shortwave_radiation, longwave_radiation
from wind_utility import windcoef


class Microclimate:
    '''
    This class encapsulates methods for computing the local variation of microclimate. It calculates various parameters such as shortwave radiation, lapserate temperature correction, longwave radiation, and sheltered wind speed for each point in the area of interest.
    
    Inputs:
    
     - year (float): The year measured by a smartphone located in the area of interest.
     - month (float): The month measured by a smartphone located in the area of interest.
     - day (float): The day measured by a smartphone located in the area of interest.
     - localtime (float): The hour measured by a smartphone located in the area of interest.
     - lat (float): The latitude of the center of the DSM.
     - long (float): The longitude of the center of the DSM.
     - dsm (np.ndarray): The digital surface model, represented as a 2D numpy matrix with elevation in each point.
     - slope (np.ndarray): The slope of the DSM, represented as a 2D numpy matrix with the slope of each point.
     - aspect (np.ndarray): The aspect of the DSM, represented as a 2D numpy matrix with the aspect of each point.
     - ha (np.ndarray): The mean horizon angle.
     - svf (np.ndarray): The sky view factor.
     - resolution (float): The DSM resolution scale.
     - study_area_coordinates (list): The coordinates of the study area.
     - dir (float): The direct radiation on a horizontal surface (MJ/m**2/hour).
     - dif (float): The diffuse radiation on a horizontal surface (MJ/m**2/hour).
     - hs (float): The specific humidity.
     - p (float): The atmospheric pressure (Pa).
     - n (float): The fractional cloud cover.
     - t_ref (float): The reference temperature (°C).
     - wind_speed (float): The wind speed at 1m height (m/s).
     - wind_direction (float): The direction from which the wind is blowing in decimal degrees.
     - alb (float): The albedo of the area of interest.
     - albr (float): The albedo of the area of interest, used for the reflected diffuse radiation.
    Attributes:
    
     - ShortWaveRad (np.ndarray): A 2D array containing the shortwave radiation for each point of the study area.
     - Tref (np.ndarray): A 2D array containing the reference temperature corrected with the lapserate rate for each point of the study area.
     - Tref_sensors (np.ndarray): A 2D array containing the average sensor temperature corrected with the lapserate rate for each point of the study area.
     - LongWaveRad (np.ndarray): A 2D array containing the longwave radiation for each point of the study area.
     - Wind (np.ndarray): A 2D array containing the wind speed at 1m height corrected with the sheltering coefficient for each point of the study area.
     - Slope (np.ndarray): The slope of the DSM.
     - Aspect (np.ndarray): The aspect of the DSM.
     - Altitude (np.ndarray): The DSM elevation.
    '''
    
    def __init__(self,
                 year, month, day, localtime, lat, long,
                 dsm, slope, aspect, ha, svf, resolution, study_area_coordinates,
                 dir, dif, hs, p, n, t_ref, wind_speed, wind_direction, alb, albr):
       
        
        julian = julday(year=year, month=month, day=day, hour=0)
        
        #compute shortwave radiation
        self.ShortWaveRad = shortwave_radiation(dir = dir, dif = dif,
                                                julian = julian, localtime = localtime, lat = lat, long = long,
                                                dsm = dsm, slope = slope, aspect = aspect,
                                                reso = resolution, ha = ha, svf = svf, 
                                                alb = alb, albr = albr)
        
        #box coordinates to select the study area from the dsm
        box_a3_lat = study_area_coordinates[0]#804
        box_a1_lat = study_area_coordinates[1]#619
        box_a3_lon = study_area_coordinates[2]#422
        box_a2_lon = study_area_coordinates[3]#588
        
        #compute lapserate correction to reference temperature
        lapserate_correction = lapserate(t_ref, hs, p)*(dsm-np.mean(dsm[box_a1_lat:box_a3_lat, box_a3_lon:box_a2_lon]))
        tc_ref = t_ref + lapserate_correction
        self.Tref = tc_ref

        #compute longwave radiation
        self.LongWaveRad = longwave_radiation(h = hs, tc = tc_ref, n = n, p = p, svf = svf, dsm_shape = dsm.shape)

        #compute wind             
        ws = windcoef(dsm, wind_direction, reso = resolution, hgt = 1)
        self.Wind = ws*wind_speed  
        
        
        self.Slope = slope 
        self.Aspect = aspect 
        self.Altitude = dsm

            
