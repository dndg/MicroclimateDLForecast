import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
This Python module elaborates the collected sensor data by performing various operations. Its central method is called 'load_clean_sensor_data', which reads the data stored in a CSV file and processes it into a Pandas dataframe. Another method, 'calibrate_sensors', calculates the calibration parameters. Lastly, the method 'elaborate_data_sensors' combines all the necessary operations and returns a Pandas dataframe containing the calibrated sensor data for the specified period.
'''



def load_clean_sensor_data(sensor_idx, filename,
                           start_period = (11,11), end_period = (1,15),hourly_av = True,
                           time_varname='Date', t_varname = 'Temp', h_varname = 'Umidità',
                           drop_cols = ['Date', 'Remark'], idx_T_rem = -2, idx_H_rem = -1
                          ):
    '''
    This method reads sensor data from a CSV file, performs cleaning and processing operations, and returns a Pandas dataframe.
    Inputs:
     - sensor_idx: The index of the sensor (sensors are numbered from 1 to 25).
     - filename: The name of the CSV file that stores the data.
     - start_period: The starting date of the period of interest defined as (month,day); (default: (11,11)). 
     - end_period: The ending date of the period of interest defined as (month,day); (default: (1,15)). 
     - hourly_av: A boolean parameter that determines whether to average the data on an hourly time scale (default: True).
     - time_varname: The name of the column in the CSV file that contains the time recordings (default: 'Date'). 
     - t_varname: The name of the column in the CSV file that contains the temperature recordings (default: 'Temp').
     - h_varname: The name of the column in the CSV file that contains the humidity recordings (default: 'Umidità').
     - drop_cols: A list of column names in the CSV file to drop at the end of the processing (default: ['Date', 'Remark']).
     - idx_T_rem: The index at which the temperature recordings may contain symbols (e.g., units [°C]) to be removed (default: -2).
     - idx_H_rem: The index at which the humidity recordings may contain symbols (e.g., units [%]) to be removed (default: -1).
    Output:
     - dfs: Pandas dataframe containing the elaborated sensor data.

    '''

    #load sensor data                              
    dfs = pd.read_csv(filename)
                              
    #create new columns with time coordinates                          
    dfs['Year'] = pd.to_datetime(dfs[time_varname]).dt.year
    dfs['Month'] = pd.to_datetime(dfs[time_varname]).dt.month
    dfs['Day'] = pd.to_datetime(dfs[time_varname]).dt.day
    dfs['Hour'] = pd.to_datetime(dfs[time_varname]).dt.hour
    dfs['Minute'] = pd.to_datetime(dfs[time_varname]).dt.minute

    #drop useless columns and rename temperature and humidity recordings                          
    dfs.drop(columns = drop_cols, inplace = True)
    dfs.rename(columns={t_varname:'Temperature', h_varname: 'Humidity'}, inplace = True)

    #transform the recordings to float numbers                          
    dfs['Temperature'] = [float(i[:idx_T_rem].replace(',','.')) for i in dfs['Temperature'].values]  
    dfs['Humidity'] = [float(i[:idx_H_rem].replace(',','.')) for i in dfs['Humidity'].values] 

    #average data by hour                          
    if hourly_av:
        dfs=dfs.groupby(by=['Year','Month','Day','Hour'],as_index=False)[['Temperature','Humidity']].mean()

    #select data for the period of interest                          
    dfs=dfs[(dfs.Month!=start_period[0])|(dfs.Day>start_period[1])]
    dfs=dfs[(dfs.Month!=end_period[0])|(dfs.Day<=end_period[1])]
                              
    #save the sensor idx                          
    dfs['Sensor_id'] =  sensor_idx
                              
    return dfs




def calibrate_sensors(filename_input,sensors_calib_idx, start_period, end_period, hourly_av, idx_T_rem, idx_H_rem,
                      time_varname='time', t_varname = 'temperature(C)', h_varname = 'humidity',drop_cols = ['time', 'note'],
                     ):
    '''
    This method calibrates the sensor recordings by calculating the calibration parameter. It uses a specific recording period during which the sensors measured humidity and temperature under controlled conditions to eliminate systematic errors. The calibration process involves computing the average of all sensor recordings for each time instance. Then, the difference between this average value and the individual sensor recordings is calculated for each sensor. Finally, the time average of these differences is determined as the calibration parameter, which can be used to adjust the sensor recordings. The present method makes also a plot of the sensors recordings previous and after the calibration.
    Inputs:
     - filename_input: The name of the CSV file that stores the data collected for the calibration period.
     - sensors_calib_idx: The index of the sensor (sensors are numbered from 1 to 25).
     - start_period: The starting date of the calibration period of interest defined as (month,day). 
     - end_period: The ending date of the calibration period of interest defined as (month,day). 
     - hourly_av: A boolean parameter that determines whether to average the data on an hourly time scale.
     - idx_T_rem: The index at which the temperature recordings may contain symbols (e.g., units) to be removed.
     - idx_H_rem: The index at which the humidity recordings may contain symbols (e.g., units) to be removed.
     - time_varname: The name of the column in the CSV file that contains the time recordings (default: 'time'). 
     - t_varname: The name of the column in the CSV file that contains the temperature recordings (default: 'temperature(C)').
     - h_varname: The name of the column in the CSV file that contains the humidity recordings (default: 'humidity').
     - drop_cols: A list of column names in the CSV file to drop at the end of the processing (default: ['time', 'note']).

    Output:
     - calib_dict: dictionary containing as keys the sensor names and as values a tuple with the calibration parameter for temperature         and humidity.
    '''

                         
    calib_dict = {}
    
    T_arr = []
    H_arr = []

    # define the figure and axis for the plot
    fig, axs = plt.subplots(2,2, figsize = (16,10),sharex=True)
    
    ax1 = axs[0,0]
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Without calibration')
    
    ax2 = axs[0,1]
    ax2.set_ylabel('Temperature [°C]')
    ax2.set_title('With calibration')
    
    ax3 = axs[1,0]
    ax3.set_ylabel('Humidity %')
    ax3.set_xlabel('Days')
    ax3.set_title('Without calibration')
    
    ax4 = axs[1,1]
    ax4.set_ylabel('Humidity %')
    ax4.set_xlabel('Days')
    ax4.set_title('With calibration')

    #load and elaborate sensors data to compute their average                     
    for sensor_idx in sensors_calib_idx:
        filename = filename_input.format(sensor_idx)
        dfs = load_clean_sensor_data(sensor_idx, filename,start_period = start_period, end_period = end_period, hourly_av = hourly_av,
                                     time_varname=time_varname, t_varname = t_varname, h_varname = h_varname,
                                     drop_cols = drop_cols,idx_T_rem = idx_T_rem, idx_H_rem =idx_H_rem)
        
        
        T_arr.append(dfs.Temperature.values)
        H_arr.append(dfs.Humidity.values)

        t = np.arange(len(dfs))/24/6 #freq registration is 6 for hour
        ax1.plot(t, dfs.Temperature.values)
        ax3.plot(t, dfs.Humidity.values)
    
    T_av_arr = np.mean(T_arr,axis=0)
    H_av_arr = np.mean(H_arr,axis=0)
    
    
    
    #compute the calibration parameter for each sensor
    for sensor_idx in sensors_calib_idx:
        filename = filename_input.format(sensor_idx)
        dfs = load_clean_sensor_data(sensor_idx, filename,start_period = start_period, end_period = end_period, hourly_av = hourly_av,
                                     time_varname=time_varname, t_varname = t_varname, h_varname = h_varname,
                                     drop_cols = drop_cols,idx_T_rem = idx_T_rem, idx_H_rem =idx_H_rem)
    
        Ts = dfs.Temperature.values
        T_calib_val = np.mean(Ts-T_av_arr)
        dfs['Temperature'] = Ts-T_calib_val
    
        Hs = dfs.Humidity.values
        H_calib_val = np.mean(Hs-H_av_arr)
        dfs['Humidity'] = Hs-H_calib_val
    
        calib_dict['sensor_{}'.format(sensor_idx)] = (T_calib_val,H_calib_val)

        t = np.arange(len(dfs))/24/6 #freq registration is 6 for hour
        ax2.plot(t, dfs.Temperature.values)
        ax4.plot(t, dfs.Humidity.values)

    return calib_dict   





def elaborate_data_sensors(filename_calibration_sensors, filename_data_sensors, sensors_calib_idx, sensors_idx,
                           Boaga_Gauss_N_list, Boaga_Gauss_E_list,
                           start_period_calibration, end_period_calibration, start_period, end_period,
                           idx_T_rem_calib, idx_H_rem_calib,
                           time_varname_calib='time', t_varname_calib = 'temperature(C)',h_varname_calib = 'humidity',
                           drop_cols_calib = ['time', 'note'],
                           time_varname_data ='Date', t_varname_data = 'Temp', h_varname_data = 'Umidità',
                           drop_cols_data = ['Date', 'Remark'], idx_T_rem_data = -2, idx_H_rem_data = -1                           
                          ):
    '''
    This method combines all the necessary operations to load, elaborate and calibrate sensors data and returns a Pandas dataframe containing the calibrated sensor data for the specified period.
    Inputs:
     - filename_calibration_sensors: The name of the CSV file that stores the data collected for the calibration period.
     - filename_data_sensors: The name of the CSV file that stores the data.
     - sensors_calib_idx: The index of the sensors which have taken measure for the calibration (sensors are numbered from 1 to 25).
     - sensors_idx: The index of the sensors (sensors are numbered from 1 to 25).
     - Boaga_Gauss_N_list: The list with all the North Gauss-Boaga coordinates of the 25 sensors.
     - Boaga_Gauss_E_list: The list with all the East Gauss-Boaga coordinates of the 25 sensors.
     - start_period_calibration: The starting date of the calibration period of interest defined as (month,day). 
     - end_period_calibration: The ending date of the calibration period of interest defined as (month,day). 
     - start_period: The starting date of the period of interest defined as (month,day)
     - end_period: The ending date of the period of interest defined as (month,day)
     - idx_T_rem_calib: The index at which the temperature recordings may contain symbols (e.g., units) to be removed.
     - idx_H_rem_calib: The index at which the humidity recordings may contain symbols (e.g., units) to be removed.
     - time_varname_calib: The name of the column in the CSV file that contains the calibration time recordings (default: 'time'). 
     - t_varname_calib: The name of the column in the CSV file that contains the calibration temperature recordings 
       (default: 'temperature(C)').
     - h_varname_calib: The name of the column in the CSV file that contains the calibration humidity recordings (default: 'humidity').
     - drop_cols_calib: A list of column names in the CSV file to drop at the end of the processing for the calibration period                 (default: ['time', 'note']).
     - time_varname_data: The name of the column in the CSV file that contains the time recordings (default: 'Date'). 
     - t_varname_data: The name of the column in the CSV file that contains the temperature recordings (default: 'Temp').
     - h_varname_data: The name of the column in the CSV file that contains the humidity recordings (default: 'Umidità').
     - drop_cols_data: A list of column names in the CSV file to drop at the end of the processing (default: ['Date', 'Remark']).
     - idx_T_rem_data: The index at which the temperature recordings may contain symbols (e.g., units) to be removed (default: -2).
     - idx_H_rem_data: The index at which the humidity recordings may contain symbols (e.g., units) to be removed (default: -1).
    Outputs:
     - df_sensors: Pandas dataframe containing all the elaborated sensors data.
    '''

                               
    #compute calibration parameters                        
    calib_dict = calibrate_sensors(filename_input = filename_calibration_sensors, sensors_calib_idx = sensors_calib_idx,
                                   start_period = start_period_calibration, end_period = end_period_calibration,
                                   hourly_av = False, idx_T_rem = idx_T_rem_calib, idx_H_rem = idx_H_rem_calib,
                                   time_varname=time_varname_calib, t_varname = t_varname_calib,h_varname = h_varname_calib,
                                   drop_cols = drop_cols_calib,  
                                  )

    
    #load sensors data for the period of interest and calibrate them
    df_list = []
                               
    for sensor_idx in sensors_idx:
        filename = filename_data_sensors.format(sensor_idx)

        dfs = load_clean_sensor_data(sensor_idx = sensor_idx, filename = filename,
                                     start_period = start_period, end_period = end_period,
                                     hourly_av = True,
                                     time_varname=time_varname_data, t_varname = t_varname_data, h_varname = h_varname_data,
                                     drop_cols = drop_cols_data, idx_T_rem = idx_T_rem_data, idx_H_rem = idx_H_rem_data
                                    )
        
        T_calib_val = calib_dict['sensor_{}'.format(sensor_idx)][0]
        H_calib_val = calib_dict['sensor_{}'.format(sensor_idx)][1]
        
        dfs['Temperature'] = dfs.Temperature.values-T_calib_val
        dfs['Humidity'] = dfs.Humidity.values-H_calib_val
    
        #add sensors location through gauss boaga coordinates
        dfs['Gauss_Boaga_N'] = Boaga_Gauss_N_list[sensor_idx-1]
        dfs['Gauss_Boaga_E'] = Boaga_Gauss_E_list[sensor_idx-1]
    
        df_list.append(dfs)

    #concatenate all sensors data in a pandas dataframe                           
    df_sensors = pd.concat(df_list)

    return df_sensors