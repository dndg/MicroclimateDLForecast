import numpy as np
import pandas as pd



def julday(year, month, day, hour):   
    '''
    Converts the time to julian day. We set the time zone to CET (central european time) which is the one of Italy.
    The conversion has been tested with online converters.
    Inputs:
     - year (float)
     - month (float)
     - day (float)
     - hour (float)
    Outputs:
     - julian date (float)
    ''' 
    
    ts = pd.Timestamp(year = year,  month = month, day = day, 
                  hour = hour)
    return ts.to_julian_date()


def solartime(localtime, long, julian):
    '''
    Calculates the local solar time (form Milner, https://en.wikipedia.org/wiki/Equation_of_time).
    Inputs:
     - localtime (float): The hour measured by a smartphone located in the area of interest.
     - long (float): The longitude of the center of the dtm.
     - julian (float): The julian date
    Outputs:
     - lst (float): The local solar time
    '''
    
    M = 6.24004077 + 0.01720197 * (julian -  2451545)
    eot = -7.659 * np.sin(M) + 9.863 * np.sin (2 * M + 3.5932)
    
    lstm = int(long/15)*15
    tc = 4*(long-lstm)+eot 
    
    lst = localtime + tc/60 
    
    return lst



def solalt(localtime, lat, long, julian):
    
    '''
    Calculates the solar altitude. Follow https://solarsena.com/solar-elevation-angle-altitude/ and for the 
    declination angle with julian days https://www.sciencedirect.com/topics/engineering/solar-declination.
    Inputs:
     - localtime (float): The hour measured by a smartphone located in the area of interest.
     - lat (float): The latitude of the center of the dtm.
     - long (float): The longitude of the center of the dtm.
     - julian (float): The julian date
    Outputs:
     - solar altitude (float)
    '''
    
    stime = solartime(localtime, long, julian)
    solar_hour_angle = 15*np.pi/180 * (stime - 12) 
    n_days = pd.to_datetime(julian, unit = 'D', origin = 'julian').day_of_year
    declination = 23.5 *np.pi/180 * np.cos(360/365*(n_days-172)*np.pi/180)
    
    sin_a = np.sin(declination)*np.sin(lat*np.pi/180)+np.cos(declination)*np.cos(lat*np.pi/180)* np.cos(solar_hour_angle)
     
    return np.arcsin(sin_a)



def solazi(localtime, lat, long, julian):
    
    '''
    Calculates the solar azimuth by https://www.pveducation.org/pvcdrom/properties-of-sunlight/azimuth-angle.
    Inputs:
     - localtime (float): The hour measured by a smartphone located in the area of interest.
     - lat (float): The latitude of the center of the dtm.
     - long (float): The longitude of the center of the dtm.
     - julian (float): The julian date
    Outputs:
     - solar azimuth (float)
    '''
    
    stime = solartime(localtime, long, julian)
    solar_hour_angle = 15*np.pi/180 * (stime - 12) 
    n_days = pd.to_datetime(julian, unit = 'D', origin = 'julian').day_of_year
    declination = 23.5 *np.pi/180 * np.cos(360/365*(n_days-172)*np.pi/180)
    
    cos_azi = (np.sin(declination)*np.cos(lat*np.pi/180)-np.cos(declination)*np.sin(lat*np.pi/180)*np.cos(solar_hour_angle))/np.cos(solalt(localtime, lat, long, julian))
        
    if solar_hour_angle>0:
        return 2*np.pi-np.arccos(cos_azi)
    
    else:
        return np.arccos(cos_azi)
    
    
    

 