import numpy as np
import pandas as pd
import sys

from data_era5_elaboration import elaborate_data_era5

USER_PATH = '/Users/marcozanchi/Work/Grade/'
LIB_PATH = USER_PATH + 'microclima/lib/'
sys.path.append(LIB_PATH + 'physical_model')
from utility import compute_specific_from_relative_humidity




def clean_data(x):
    '''
    Cleans ARPA station data, replacing empty recordings with the average between the previous and the next recordings if available.
    Inputs:
     - x (np.ndarray): The numpy array which contains the data to clean.
    Outputs:
     - x (np.ndarray): The cleaned numpy array.
    '''
    for idx, el in enumerate(x):
        if el == -999 and x[idx-1]!=-999 and x[idx+1]!=-999:
            x[idx] = np.mean([x[idx-1],x[idx+1]])
        elif el == -999 and x[idx-1]==-999 and x[idx+1]!=-999:
            x[idx] = x[idx+1]
        elif el == -999 and x[idx-1]!=-999 and x[idx+1]==-999:
            x[idx] = x[idx-1]
        elif el == -999 and x[idx-1]==-999 and x[idx+1]==-999:
            print('Higly corrupted data')
            
    return x
    

def load_arpa_data(variable, file_ARPA, rename_var= " Medio"):
    '''
    Loads and cleans the data of a specific variable collected by an ARPA station from a CSV file.
    Inputs:
     - variable (str): The name of the physical variable recorded by the ARPA station and saved in the file_ARPA
     - file_ARPA (str): The name of the file.csv containing the data collected by the ARPA station
     - rename_var (str): The name of the original column of the file_ARPA (default: " Medio")
    Output:
     - df (pd.dataframe): The pandas dataframe containing the cleaned data recorded from ARPA station for a specific variable.
    '''

    
    df = pd.read_csv(file_ARPA)
    df['Year'] = pd.to_datetime(df['Data-Ora']).dt.year
    df['Month'] = pd.to_datetime(df['Data-Ora']).dt.month
    df['Day'] = pd.to_datetime(df['Data-Ora']).dt.day
    df['Hour'] = pd.to_datetime(df['Data-Ora']).dt.hour
    df = df.rename(columns={rename_var: variable})
    df[variable] = clean_data(df[variable].values)
    
    return df


def elaborate_data_arpa(foldername, year, month, days_range):
    '''
    Loads and elaborates all the data collected from ARPA station (temperature, precipitation, relative humidity, wind speed and wind direction) for a defined period.
    Inputs:
     - foldername  (str): The name of path to the folder containing the ARPA data for a specific period.
     - year (float): The year of interest.
     - month (float): The month of interest.
     - days_range ((float,float)): The range of days [day_start, day_end) of interest.
    Output:
     - df_T (pd.dataframe): The pandas dataframe containing all the data collected by the ARPA station. 
    '''

    file_T = foldername+'temperature.csv'
    df_T = load_arpa_data("Temperature", file_T, " Medio")
    df_T = df_T[(df_T.Year == year)&(df_T.Month == month)&(df_T.Day >= days_range[0])&(df_T.Day < days_range[1])]
    
    file_Prec = foldername+'precipitation.csv'
    df_Prec = load_arpa_data("Precipitation", file_Prec, "Valore Cumulato")
    df_Prec = df_Prec[(df_Prec.Year == year)&(df_Prec.Month == month)&(df_Prec.Day >= days_range[0])&(df_Prec.Day < days_range[1])]
    
    file_Ws = foldername+'wind_speed.csv'
    df_Ws = load_arpa_data("Wind speed", file_Ws, " Medio")
    df_Ws = df_Ws[(df_Ws.Year == year)&(df_Ws.Month == month)&(df_Ws.Day >= days_range[0])&(df_Ws.Day < days_range[1])]
    
    file_Wd = foldername+'wind_direction.csv'
    df_Wd = load_arpa_data("Wind direction", file_Wd, " Medio")
    df_Wd = df_Wd[(df_Wd.Year == year)&(df_Wd.Month == month)&(df_Wd.Day >= days_range[0])&(df_Wd.Day < days_range[1])]
    
    file_Hrel = foldername+'relative_humidity.csv'
    df_Hrel = load_arpa_data("Relative humidity", file_Hrel, " Medio")
    df_Hrel = df_Hrel[(df_Hrel.Year == year)&(df_Hrel.Month == month)&(df_Hrel.Day >= days_range[0])&(df_Hrel.Day < days_range[1])]

    df_T['Precipitation'] = df_Prec.Precipitation.values
    df_T['Wind speed'] = df_Ws['Wind speed'].values
    df_T['Wind direction'] = df_Wd['Wind direction'].values
    df_T['Relative humidity'] = df_Hrel['Relative humidity'].values

    return df_T


def elaborate_data_mixed_arpa_era5(era5_grib_filename, arpa_foldername,dsm_study_case,arpa_station_altitude,
                                   year, month, days_range):
    '''
    Loads and elaborates the data collected by ERA5 database and by an ARPA station for a defined period and than merges them.
    Inputs:
     - era5_grib_filename (str): The file downloaded from the ERA5 database in GRIB format.
     - arpa_foldername (str): The name of path to the folder containing the ARPA data for a specific period.
     - dsm_study_case (np.ndarray): the dsm of the study case area
     - arpa_station_altitude (float): the altitude of the ARPA station
     - year (float): The year of interest.
     - month (float): The month of interest.
     - days_range ((float,float)): The range of days [day_start, day_end) of interest.
    Output:
     - df_era5 (pd.dataframe): The pandas dataframe containing all the merged data collected by the ARPA station and by ERA5 database. 
     
    '''
                                       

    df_era5 = elaborate_data_era5(grib_filename = era5_grib_filename, year = year, month = month, days_range = days_range)                                  
    df_arpa = elaborate_data_arpa(foldername = arpa_foldername,
                                  year = year, month = month, days_range = days_range)

    assert len(df_era5) == len(df_arpa)
    assert np.sum(df_era5.Hour.values == df_arpa.Hour.values) == len(df_era5)                                     
    assert np.sum(df_era5.Day.values == df_arpa.Day.values) == len(df_era5)  
    assert np.sum(df_era5.Month.values == df_arpa.Month.values) == len(df_era5)  
    assert np.sum(df_era5.Year.values == df_arpa.Year.values) == len(df_era5)  
                                       
    df_era5.loc[:,'Temperature 1m'] = df_arpa["Temperature"].values-(np.mean(dsm_study_case)-arpa_station_altitude)*6.5/1000
    df_era5.loc[:,'Precipitation'] = df_arpa["Precipitation"].values/1000
    df_era5.loc[:,'Wind speed 1m'] = df_arpa['Wind speed'].values
    df_era5.loc[:,'Wind direction'] = df_arpa['Wind direction'].values
    df_era5.loc[:,'Relative humidity'] = df_arpa["Relative humidity"].values
    df_era5.loc[:,'Specific humidity'] = compute_specific_from_relative_humidity(df_era5['Relative humidity'].values,
                                                                            df_era5['Pressure'].values, 
                                                                            df_era5['Temperature 1m'].values)

    return df_era5    