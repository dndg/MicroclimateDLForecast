import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys

import tensorflow as tf
from tensorflow.keras.models import Model

USER_PATH = '/Users/marcozanchi/Work/Grade/'
LIB_PATH = USER_PATH + 'microclima/lib/'

sys.path.append(LIB_PATH + 'physical_model')
from microclimate import Microclimate



def make_microclimate_prediction_over_study_area(model, train_data_rescale_params,
                                                 df_climate_micro, year, month, day, hour,
                                                 lat, long, dsm, resolution, slope, aspect,ha, svf,study_area_coordinates):

        dir = df_climate_micro['Direct shortwave radiation'].values[0]
        dif = df_climate_micro['Diffuse shortwave radiation'].values[0]
        
        t_ref = df_climate_micro['Temperature 1m'].values[0]
        p = df_climate_micro['Pressure'].values[0]
        tdew =  df_climate_micro['Dew Temperature 1m'].values[0]
        hs = df_climate_micro['Specific humidity'].values[0]
        hr_ref = df_climate_micro['Relative humidity'].values[0] #not used for the microclimate model
        n = df_climate_micro['Cloud cover'].values[0]
        wind_direction = df_climate_micro['Wind direction'].values[0]
        wind_speed = df_climate_micro['Wind speed 1m'].values[0]
        
        alb = 0.25
        albr = 0.25
        
        micro = Microclimate(year = year, month = month, day = day, localtime = hour,
                             lat = lat, long = long,
                             dsm = dsm, slope = slope, aspect = aspect, ha = ha, svf = svf,
                             resolution = resolution, study_area_coordinates = study_area_coordinates,
                             dir = dir, dif = dif, hs = hs, p = p, n = n, t_ref = t_ref,
                             wind_speed = wind_speed, wind_direction = wind_direction,
                             alb = alb, albr = albr)

        box_a3_lat = study_area_coordinates[0]
        box_a1_lat = study_area_coordinates[1]
        box_a3_lon = study_area_coordinates[2]
        box_a2_lon = study_area_coordinates[3]

        shortwave_rad_out = micro.ShortWaveRad[box_a1_lat:box_a3_lat, box_a3_lon:box_a2_lon]
        longwave_rad_out = micro.LongWaveRad[box_a1_lat:box_a3_lat, box_a3_lon:box_a2_lon]
        wind_out = micro.Wind[box_a1_lat:box_a3_lat, box_a3_lon:box_a2_lon]
        T_ref_out = micro.Tref[box_a1_lat:box_a3_lat, box_a3_lon:box_a2_lon]
        H_ref_out = hr_ref
        precipitation_out = df_climate_micro.Precipitation.values[0]
        pressure_out = p
        cloud_cover_out = n
        specific_humidity_out = hs

        Inputs = []
        for idx_row in range(len(shortwave_rad_out)):
            for idx_col in range(len(shortwave_rad_out[0])):

                #Pay attention to the order of the inputs!!! It must concide with the input_variables order
                Input = np.array(
                         [shortwave_rad_out[idx_row,idx_col],longwave_rad_out[idx_row,idx_col],
                          wind_out[idx_row,idx_col],
                          T_ref_out[idx_row,idx_col], H_ref_out,
                          precipitation_out, pressure_out, cloud_cover_out, specific_humidity_out]
                                )
                Input = (Input-train_data_rescale_params['mean'])/train_data_rescale_params['std']
                Inputs.append(Input)

        microclima_map = model.predict(np.array(Inputs),verbose = False)
        microclima_map = microclima_map.reshape(len(shortwave_rad_out)*len(shortwave_rad_out[0]))
        microclima_map = microclima_map.reshape((len(shortwave_rad_out),len(shortwave_rad_out[0])))
            
        return microclima_map
        
        
        
    

