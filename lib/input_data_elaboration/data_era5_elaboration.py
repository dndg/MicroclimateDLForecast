import numpy as np
import pandas as pd
import sys
import math
import pygrib


USER_PATH = '/Users/marcozanchi/Work/Grade/'
LIB_PATH = USER_PATH + 'microclima/lib/'
sys.path.append(LIB_PATH + 'physical_model')
from wind_utility import windheight
from utility import compute_specific_humidity,compute_relative_humidity




def load_data_era5(grib_filename, year, month, days_range=(1,32),study_area_loc=(7,7)):
    '''
    This method retrieves data from the ERA5 database for a specific period and study area location, storing it in a Pandas dataframe.
    Inputs:
     - grib_filename:  The file downloaded from the ERA5 database in GRIB format.
     - year: The year of the period of interest.
     - month: The month of the period of interest
     - days_range: The range of days for the period of interest (start day, end day).
     - study_area_loc: The coordinates (row, column) of the study area location in the ERA5 grid.
    Output:
     - metadata_df: A Pandas dataframe containing the climate data for each hour of each day within the period of interest.
    '''


    grbs = pygrib.open(grib_filename)
    var_list = ['10 metre U wind component',
            '10 metre V wind component',
            '2 metre dewpoint temperature',
            '2 metre temperature',
            'Total sky direct solar radiation at surface',
            'Surface solar radiation downwards',
            'Total precipitation',
            'Total cloud cover',
            'Surface pressure'
            ]

    days = np.arange(days_range[0],days_range[1],1)
    hours = np.arange(0,24,1)
    
    variables_arr = []

    for var_name in var_list:
        variables_appo = [] 
        grb = grbs.select(name = var_name)

        day_arr = []
        hour_arr = []
        month_arr = []
        year_arr = []
        for day in days:
            for hour in hours:
                idx_day_hour = (day-days_range[0])*24+hour
                data = grb[idx_day_hour].values.data
                data_TB = data[study_area_loc[0],study_area_loc[1]] 
                variables_appo.append(data_TB)

                day_arr.append(day)
                hour_arr.append(hour)
                month_arr.append(month)
                year_arr.append(year)

        variables_arr.append(variables_appo)
    
    
    metadata_dict = {
                    'day': day_arr,
                    'hour': hour_arr,
                    'month': month_arr,
                    'year':year_arr,
                    '10 metre U wind component': variables_arr[0],
                    '10 metre V wind component': variables_arr[1],
                    '2 metre dewpoint temperature': variables_arr[2],
                    '2 metre temperature': variables_arr[3],
                    'Total sky direct solar radiation at surface': variables_arr[4],
                    'Surface solar radiation downwards': variables_arr[5],
                    'Total precipitation': variables_arr[6],
                    'Total cloud cover': variables_arr[7],
                    'Surface pressure': variables_arr[8]
                    }    
    
    
    metadata_df = pd.DataFrame(metadata_dict)

    return metadata_df






def elaborate_data_era5(grib_filename, year, month, days_range=(1,32),study_area_loc=(7,7)):
    '''
    This method retrieves data from the ERA5 database for a specific period and study area location, and processes it to generate an elaborated climate dataset. The processing includes calculating wind speed and direction at a height of 1 meter, rescaling radiation variables to units of [MJ/m**2/hour], and rescaling temperature variables to a height of 1 meter.
    Inputs:
     - grib_filename:  The file downloaded from the ERA5 database in GRIB format.
     - year: The year of the period of interest.
     - month: The month of the period of interest
     - days_range: The range of days for the period of interest (start day, end day).
     - study_area_loc: The coordinates (row, column) of the study area location in the ERA5 grid.
    Output:
     - metadata_df: A Pandas dataframe containing the elaborated climate data for each hour of each day within the period of interest.
    '''
    
    metadata_df = load_data_era5(grib_filename, year, month, days_range,study_area_loc)
    metadata_listdict = []
    for _,row in metadata_df.iterrows():

        wspeed10 = np.sqrt(row['10 metre U wind component']**2 + row['10 metre V wind component']**2)
        wdir = (math.atan2(row['10 metre U wind component'], row['10 metre V wind component']) * 180/np.pi + 180)%360

        dri = row['Total sky direct solar radiation at surface'] * 0.000001 

        T_1m = (row['2 metre temperature'] - 273.15)+6.5/1000
        Tdew_1m = (row['2 metre dewpoint temperature'] - 273.15)+6.5/1000

        h_spec = compute_specific_humidity(Tdew_1m, T_1m, row['Surface pressure'])
        h_rel = compute_relative_humidity(Tdew_1m, T_1m)

        metadata_dict = {
                        'Day': row.day,
                        'Hour': row.hour,
                        'Month': row.month,
                        'Year': row.year,
                        'Temperature 1m': T_1m,
                        'Dew Temperature 1m': Tdew_1m,
                        'Specific humidity':h_spec,
                        'Relative humidity':h_rel,
                        'Pressure': row['Surface pressure'],
                        'Cloud cover': row['Total cloud cover'],
                        'Wind speed 1m': windheight(wspeed10, 10, 1),
                        'Wind direction': wdir,
                        'Direct shortwave radiation': dri,
                        'Diffuse shortwave radiation': row['Surface solar radiation downwards']* 0.000001  - dri,
                        'Precipitation': row['Total precipitation']
                        }

        metadata_listdict.append(metadata_dict)
    
    return pd.DataFrame(metadata_listdict)
    