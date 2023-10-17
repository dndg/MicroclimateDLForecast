import numpy as np
import pandas as pd



def load_data(df_filename):
    """
    Loads a DataFrame from a pickle file.
    
    Parameters:
        df_filename (str): The filename of the pickle file.
        
    Returns:
        df (DataFrame): The loaded DataFrame.
    """
    
    df = pd.read_pickle(df_filename)
    return df




def split_train_val_test(df, seed, test_train_proportion = 0.3, val_test_proportion = 0.3):
    
    """
    Split DataFrame into Train, Validation, and Test Datasets.
    Randomly splits a DataFrame into train, validation, and test datasets based on the given proportions and on the specific sensor's ID.
    
    Parameters:
        df (DataFrame): The DataFrame to be split.
        seed (int): The seed value for randomization.
        test_train_proportion (float): The proportion of data to allocate for the test and train datasets.
                                       Default: 0.3.
        val_test_proportion (float): The proportion of data to allocate for the validation and test datasets.
                                     Default: 0.3.
                                     
    Returns:
        train_df (DataFrame): The train dataset containing randomly divided data from specific sensors.
        val_df (DataFrame): The validation dataset containing randomly divided data from specific sensors.
        test_df (DataFrame): The test dataset containing randomly divided data from specific sensors.
    """
    

    #randomly mix the sensors id
    sensors = df.sample(frac=1, random_state = seed).Sensor_id.unique()

    #split sensors between train and test
    sensors_train = sensors[:-int(len(sensors)*test_train_proportion)]
    sensors_test = sensors[-int(len(sensors)*test_train_proportion):]

    #split sensors between test and validation
    sensors_val = sensors_test[-int(len(sensors_test)*val_test_proportion):]
    sensors_test =  sensors_test[:-int(len(sensors_test)*val_test_proportion)]

    #create train, test, val dataframes
    df_train = df[df.Sensor_id.isin(sensors_train)]
    df_test = df[df.Sensor_id.isin(sensors_test)]
    df_val = df[df.Sensor_id.isin(sensors_val)]

    #check that each dataframe contains different sensors
    assert not set(list(df_train.Sensor_id.values)) & set(list(df_test.Sensor_id.values))
    assert not set(list(df_train.Sensor_id.values)) & set(list(df_val.Sensor_id.values))
    assert not set(list(df_val.Sensor_id.values)) & set(list(df_test.Sensor_id.values))

    #check that all that have been represented in the dataframes
    assert len(df_train)+len(df_test)+len(df_val) == len(df)

    return df_train, df_test, df_val




def load_split_data(df_filename, seed, test_train_proportion = 0.3, val_test_proportion = 0.3):
    """
    This function loads a DataFrame from a pickle file and splits it into train, validation, and test datasets based on the specified proportions. Each dataset contains data from specific sensors, which are randomly divided according to their ID.
    
    Parameters:
        df_filename (str): The filename of the pickle file.
        seed (int): The seed value used for randomization.
        test_train_proportion (float): The proportion of data to allocate for the test and train datasets. Default: 0.3.
        val_test_proportion (float): The proportion of data to allocate for the validation and test datasets. Default: 0.3.
    
    Returns:
        train_df (DataFrame): The train dataset containing data from specific sensors randomly divided by ID.
        val_df (DataFrame): The validation dataset containing data from specific sensors randomly divided by ID.
        test_df (DataFrame): The test dataset containing data from specific sensors randomly divided by ID.
    """

    df = load_data(df_filename = df_filename)
    df_train, df_test, df_val = split_train_val_test(df = df, seed = seed,
                                                     test_train_proportion = test_train_proportion,
                                                     val_test_proportion = val_test_proportion)

    return df_train, df_test, df_val




def get_data_from_df(df, input_variables, target_variable):
    """
    This function creates input and output arrays for a neural network from a given dataset. The first array, `X`, contains the variables defined by the `input_variables` list, while the second array, `y`, contains the target variable.
    
    Parameters:
        df (DataFrame): The dataset containing the input and target data.
        input_variables (list[str]): List of column names from the DataFrame to be used as input data for the neural network.
        target_variable (str): Name of the column in the DataFrame representing the target variable.
    
    Returns:
        X (np.ndarray): Array of input variables extracted from the dataset for each data point.
        y (np.ndarray): Array of the target variable extracted from the dataset for each data point.
    """


    X = []
    y = []

    for _,row in df.iterrows():

        _X = []
        for input_variable in input_variables:
           _X.append(row[input_variable])
        _X = np.array(_X)
                     
        _y = row[target_variable]

        X.append(_X)
        y.append(_y)

    X = np.array(X)
    y = np.array(y)
    
    return X,y





