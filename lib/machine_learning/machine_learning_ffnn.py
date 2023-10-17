import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import copy
from scipy.stats import linregress
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, LSTM
from sklearn.linear_model import LinearRegression

USER_PATH = '/Users/marcozanchi/Work/Grade/'
LIB_PATH = USER_PATH + 'microclima/lib/'
RESULTS_PATH = USER_PATH + 'microclima/notebooks/results/'

sys.path.append(LIB_PATH + 'machine_learning')
from elaborate_data_utility import load_split_data, get_data_from_df
from model_utility import create_model, train_model

sys.path.append(LIB_PATH + 'plotting_methods')
from machine_learning_plots import plot_history, plot_results_test_R2, plot_distribution_test_mae, plot_inputs_importance, plot_cross_period_prediction, plot_microclimate_prediction_over_study_area

sys.path.append(LIB_PATH + 'microclima_generation')
from predict_microclimate_over_study_area import make_microclimate_prediction_over_study_area
 




class MachineLearningFFNN:
    """
    This class encapsulates all the procedures to prepare the data, train, test and evaluate a feedforward neural network to predict the temperature or the ralative humidity starting from climate data.

    Attributes:
        model_type_name (str): The name of the model type.
        model_period_name (str): The name of the model period.
        df_filename (str): The filename of the dataset.
        input_variables (list): List of input variable names.
        target_variable (str): The name of the target variable.
        max_n_neurons (int): The maximum number of neurons per layer.
        min_n_neurons (int): The minimum number of neurons per layer.
        n_layers (int): The number of layers in the neural network.
        learning_rate (float): The learning rate for the model.
        activation (str): The activation function for the model.
        loss (str): The loss function for the model.
        metrics (list): List of evaluation metrics for the model.
        save_path (str): The path to save the trained model. Default: None.
        df_train (DataFrame): The training dataset. Default: None.
        df_test (DataFrame): The test dataset. Default: None.
        df_validation (DataFrame): The validation dataset. Default: None.
        X_train (ndarray): The input data for training. Default: None.
        y_train (ndarray): The target data for training. Default: None.
        X_validation (ndarray): The input data for validation. Default: None.
        y_validation (ndarray): The target data for validation. Default: None.
        X_test (ndarray): The input data for testing. Default: None.
        y_test (ndarray): The target data for testing. Default: None.
        train_data_rescale_params (dict): The rescaling parameters for training data. Default: None.
        model (object): The trained model. Default: None.
        history_df (DataFrame): The training history of the model. Default: None.
        model_name (str): The name of the model.

    Methods:
        elaborate_input_data: Loads, splits and normalizes input data.
        elaborate_input_data_fixed_sensors: Loads, splits and normalizes input data according prefixed sensors indexes.
        create_model: Creates the model based on input hyperparameters.
        train: Train the model
        test: Computes R2 and MAE on the test set
        evaluate: Returns R2 and MAE values on the test set
        save_model_properties: Saves the datsets, hyperparameters, model weights
        make_plot_results_test_R2: Makes a scatterplot that illustrates the differences between the true values and the corresponding predicted values for each point in the test set.
        make_plot_distribution_test_mae: Plots the distribution of mean absolute errors within the test set.
        make_plot_inputs_importance: Plots the importance of input variables in a feed-forward neural network.
        make_plot_cross_period_prediction: Plots Cross-Period Prediction. The plots include the R2 scores and mean absolute errors for the cross-period predictions.
        make_plot_microclimate_prediction_over_study_area: Showcases the predicted temperature variations at a specific time instance over the entire study area.
             
    """
    
    def __init__(self,
                 model_type_name : str,
                 model_period_name: str,
                 df_filename : str,
                 input_variables : list,
                 target_variable : str,
                 max_n_neurons : int,
                 min_n_neurons : int,
                 n_layers : int,
                 learning_rate : float,
                 activation : str,
                 loss : str,
                 metrics : list,
                 save_path = None,
                 df_train = None,
                 df_test = None,
                 df_validation = None,
                 X_train = None,
                 y_train = None,
                 X_validation = None,
                 y_validation = None,
                 X_test = None,
                 y_test = None,
                 train_data_rescale_params = None,
                 model = None,
                 history_df = None
                ):
        """
        Constructor
            
        Initializes an instance of the class.
            
    
        Parameters:
            model_type_name (str): The name of the model type.
            model_period_name (str): The name of the model period.
            df_filename (str): The filename of the dataset.
            input_variables (list): List of input variable names.
            target_variable (str): The name of the target variable.
            max_n_neurons (int): The maximum number of neurons per layer.
            min_n_neurons (int): The minimum number of neurons per layer.
            n_layers (int): The number of layers in the neural network.
            learning_rate (float): The learning rate for the model.
            activation (str): The activation function for the model.
            loss (str): The loss function for the model.
            metrics (list): List of evaluation metrics for the model.
            save_path (str): The path to save the trained model. Default: None.
            df_train (DataFrame): The training dataset. Default: None.
            df_test (DataFrame): The test dataset. Default: None.
            df_validation (DataFrame): The validation dataset. Default: None.
            X_train (ndarray): The input data for training. Default: None.
            y_train (ndarray): The target data for training. Default: None.
            X_validation (ndarray): The input data for validation. Default: None.
            y_validation (ndarray): The target data for validation. Default: None.
            X_test (ndarray): The input data for testing. Default: None.
            y_test (ndarray): The target data for testing. Default: None.
            train_data_rescale_params (dict): The rescaling parameters for training data. Default: None.
            model (object): The trained model. Default: None.
            history_df (DataFrame): The training history of the model. Default: None.
        """

        self.model_type_name = model_type_name
        self.model_period_name = model_period_name                              
        self.df_filename = df_filename
        self.input_variables = input_variables
        self.target_variable = target_variable
        self.max_n_neurons = max_n_neurons
        self.min_n_neurons = min_n_neurons
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss
        self.metrics = metrics             
        self.save_path = save_path          
        self.df_train = df_train
        self.df_test = df_test
        self.df_validation = df_validation
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation            
        self.y_validation = y_validation
        self.X_test = X_test            
        self.y_test = y_test
        self.train_data_rescale_params = train_data_rescale_params
        self.model = model
        self.history_df = history_df   
        
        self.model_name = self.model_type_name + ' ' +  self.model_period_name 
                    
                    

    def elaborate_input_data(self, seed, test_train_proportion, val_test_proportion):
        """
        Loads and splits the input data in the train, validaton and test sets and normalizes them.
        
        Parameters:
            seed (int): The seed value for randomization.
            test_train_proportion (float): The proportion of data to allocate for the test and train datasets.
                                           Default: 0.3.
            val_test_proportion (float): The proportion of data to allocate for the validation and test datasets.
                                         Default: 0.3.
        """
         
        self.df_train, self.df_test, self.df_validation = load_split_data(df_filename = self.df_filename,
                                                                          seed = seed,
                                                                          test_train_proportion = test_train_proportion,
                                                                          val_test_proportion = val_test_proportion
                                                                         )
                     
        self.X_train, self.y_train = get_data_from_df(self.df_train,
                                                      input_variables = self.input_variables,
                                                      target_variable = self.target_variable)
         
        self.X_validation, self.y_validation = get_data_from_df(self.df_validation,
                                                                input_variables = self.input_variables,
                                                                target_variable = self.target_variable)
         
        self.X_test, self.y_test = get_data_from_df(self.df_test,
                                                    input_variables = self.input_variables,
                                                    target_variable = self.target_variable) 
        
        _m = np.mean(self.X_train, axis=0)
        _s = np.std(self.X_train, axis=0)
        
        self.X_train = ((self.X_train - _m) / _s)
        self.X_validation = ((self.X_validation - _m) / _s)
        self.X_test =  ((self.X_test - _m) / _s)
        
        train_data_rescale_params = {"mean": _m,"std": _s}
        
        self.train_data_rescale_params = train_data_rescale_params

        

    def elaborate_input_data_fixed_sensors(self,idxs_train, idxs_validation, idxs_test):
        """
        Loads and splits the input data in the train, validaton and test sets according to fixed index and normalizes them.
        
        Parameters:
            idxs_train (list): The list of the sensors indexes used in the training.
            idxs_validation (list): The list of the sensors indexes used in the validation.
            idxs_test (list): The list of the sensors indexes used in the testing.
        """
        df = pd.read_pickle(self.df_filename)
        self.df_train = df[df.Sensor_id.isin(idxs_train)]
        self.df_test = df[df.Sensor_id.isin(idxs_test)]
        self.df_validation = df[df.Sensor_id.isin(idxs_validation)]
    
        #check that each dataframe contains different sensors
        assert not set(list(self.df_train.Sensor_id.values)) & set(list(self.df_test.Sensor_id.values))
        assert not set(list(self.df_train.Sensor_id.values)) & set(list(self.df_validation.Sensor_id.values))
        assert not set(list(self.df_validation.Sensor_id.values)) & set(list(self.df_test.Sensor_id.values))
    
        #check that all that have been represented in the dataframes
        assert len(self.df_train)+len(self.df_test)+len(self.df_validation) == len(df)
         
                     
        self.X_train, self.y_train = get_data_from_df(self.df_train,
                                                      input_variables = self.input_variables,
                                                      target_variable = self.target_variable)
         
        self.X_validation, self.y_validation = get_data_from_df(self.df_validation,
                                                                input_variables = self.input_variables,
                                                                target_variable = self.target_variable)
         
        self.X_test, self.y_test = get_data_from_df(self.df_test,
                                                    input_variables = self.input_variables,
                                                    target_variable = self.target_variable) 
        
        _m = np.mean(self.X_train, axis=0)
        _s = np.std(self.X_train, axis=0)
        
        self.X_train = ((self.X_train - _m) / _s)
        self.X_validation = ((self.X_validation - _m) / _s)
        self.X_test =  ((self.X_test - _m) / _s)
        
        train_data_rescale_params = {"mean": _m,"std": _s}
        
        self.train_data_rescale_params = train_data_rescale_params


    def create_model(self, display_summary = True):
        """
        Create a neural network model.
    
        Parameters:
            display_summary (bool, optional): Whether to dispay the model summary. Default to True.
        """
        
        self.model = create_model(max_n_neurons = self.max_n_neurons, min_n_neurons = self.min_n_neurons,
                                  n_layers = self.n_layers, learning_rate = self.learning_rate,
                                  activation = self.activation, loss = self.loss,
                                  metrics = self.metrics, input_shape = self.X_train[0].shape)

        if display_summary:
            self.model.summary()

    

    def train(self, epochs, patience,
              metric_to_monitor = 'val_loss', restore_best_weights = True, 
              plot_history_bool = True, verbose = 1
             ):
        """
        Train a neural network model with the given training and validation data.
    
        Parameters:
            epochs (int): The number of training epochs.
            patience (int): The number of epochs with no improvement after which training will be stopped if `restore_best_weights` is set to True.
            metric_to_monitor (str, optional): The metric to be monitored for early stopping. Defaults to 'val_loss'.
            restore_best_weights (bool, optional): Whether to restore the weights of the best model during training. Defaults to True.
            plot_history_bool (bool, optional): Whether to plot the training hisotry. Default to True.
            verbose (int, optional): Verbosity mode (0 = silent, 1 = progress bar). Defaults to 1.
        """

        self.model, self.history_df = train_model(model = self.model,
                                                  X_train = self.X_train, y_train = self.y_train,
                                                  X_validation = self.X_validation, y_validation = self.y_validation,
                                                  epochs = epochs, patience = patience,
                                                  metric_to_monitor = metric_to_monitor,
                                                  restore_best_weights = restore_best_weights,
                                                  verbose = verbose
                                                 )

        if plot_history_bool:
            fig, ax = plot_history(self.history_df, title = self.model_name + ' history',
                                   color_train = "0.5",
                                   color_val =  "#ff2d55", 
                                   )
            self._figure_history = fig


            

    def test(self,):
        """
        Tests the neural network accuracy on the test set printing the R2 and MAE. It prints also the benchmark parameters.

        """
        
        self._y_test_pred = self.model.predict(self.X_test, verbose=0)
        self._y_test_pred = self._y_test_pred.reshape(len(self.y_test))
        y_bench = self.df_test.T_ref.values
        print('Benchmark vs prediction mae:',np.mean(np.abs(self.y_test-y_bench)),' ', np.mean(np.abs(self.y_test-self._y_test_pred)))
        print('Benchmark vs prediction r2:', linregress(self.y_test, y_bench).rvalue ** 2,' ', linregress(self.y_test, self._y_test_pred).rvalue ** 2)

        


    
    def evaluate(self,):
        """
        Returns the neural network accuracy on the test set printing the R2 and MAE.
        """

        mae = np.mean(np.abs(self.y_test-self._y_test_pred))
        R2 = linregress(self.y_test, self._y_test_pred).rvalue ** 2

        return mae, R2

        
        

    
    def save_model_properties(self, datetime): 
        """
        Saves all the relevant model properties in a folder within the results directory, with the folder name based on the current datetime. The saved properties include the train, validation, and test dataframes, normalization parameters, history dataframes, history figures, and the model weights. This comprehensive storage enables the reproduction of the model results for future research purposes.

        Parameters:
            datetime (str): The datetime string specifying the current time instance to save the model properties.

        """
        
        if self.target_variable == 'T_target':
            variable = 'temperature'
        elif self.target_variable == 'H_target':
            variable = 'humidity'

        if self.model_type_name == 'ERA5':
            model_type = 'era5'
        elif self.model_type_name == 'Mixed ARPA ERA5':
            model_type = 'mixed_arpa_era5'

        assert self.df_train.Year.values[0] == self.df_test.Year.values[0] == self.df_validation.Year.values[0]
        assert self.df_train.Month.values[0] == self.df_test.Month.values[0] == self.df_validation.Month.values[0]
        assert self.df_train.Day.values[0] == self.df_test.Day.values[0] == self.df_validation.Day.values[0]
        start_period = '{}_{:02d}_{:02d}'.format(self.df_train.Year.values[0], self.df_train.Month.values[0], self.df_train.Day.values[0])
        end_period = '{}_{:02d}_{:02d}'.format(self.df_train.Year.values[-1], self.df_train.Month.values[-1], self.df_train.Day.values[-1])
        
        period =  start_period + '__' + end_period

        parent_dir = RESULTS_PATH + 'output_data/results_machine_learning/models_weights/models_weights_{}/{}_{}/model_weights_{}_{}__{}/'.format(variable, variable, model_type, variable, model_type, period)

        model_dir = 'FFNN_{}_{}_{}_out_{}'.format(variable, model_type, self.model_period_name, datetime)
        self._save_path = parent_dir + model_dir
        os.mkdir(os.path.join(parent_dir, model_dir))

        #dataframes
        path = os.path.join(self._save_path, 'dataframes')
        os.mkdir(path)
        #model
        path = os.path.join(self._save_path, 'model_weights')
        os.mkdir(path)
        #figures
        path = os.path.join(self._save_path, 'figures')
        os.mkdir(path)


        self.df_train.to_pickle(self._save_path + '/dataframes/df_train.p')
        self.df_validation.to_pickle(self._save_path + '/dataframes/df_validation.p')
        self.df_test.to_pickle(self._save_path + '/dataframes/df_test.p')

        pd.to_pickle(self.train_data_rescale_params, self._save_path + '/dataframes/train_data_rescale_params.p')

        hyperparam_dict = {
                            'model_name': self.model_name,
                            'max_n_neurons':self.max_n_neurons,
                            'min_n_neurons':self.min_n_neurons,
                            'n_layers':self.n_layers,
                            'learning_rate':self.learning_rate,
                            'activation': self.activation,
                            'loss': self.loss,
                            'metrics': self.metrics,    
                          }
        pd.to_pickle(hyperparam_dict, self._save_path + '/dataframes/hyperparam_dict.p')

        self.history_df.to_pickle(self._save_path + '/dataframes/df_history.p')

        self.model.save(self._save_path + '/model_weights')

        self._figure_history.savefig(self._save_path + '/figures/figure_history.png', dpi=150)



    def load_model_properties(self, directory_path):

        self.df_train = pd.read_pickle(directory_path + '/dataframes/df_train.p')
        self.df_validation = pd.read_pickle(directory_path + '/dataframes/df_validation.p')
        self.df_test = pd.read_pickle(directory_path + '/dataframes/df_test.p')
        self.train_data_rescale_params = pd.read_pickle(directory_path + '/dataframes/train_data_rescale_params.p')

        self.X_train, self.y_train = get_data_from_df(self.df_train,
                                                      input_variables = self.input_variables,
                                                      target_variable = self.target_variable)
         
        self.X_validation, self.y_validation = get_data_from_df(self.df_validation,
                                                                input_variables = self.input_variables,
                                                                target_variable = self.target_variable)
         
        self.X_test, self.y_test = get_data_from_df(self.df_test,
                                                    input_variables = self.input_variables,
                                                    target_variable = self.target_variable) 
        
        _m = self.train_data_rescale_params['mean']
        _s = self.train_data_rescale_params['std']
        
        self.X_train = ((self.X_train - _m) / _s)
        self.X_validation = ((self.X_validation - _m) / _s)
        self.X_test =  ((self.X_test - _m) / _s)
        

        self.create_model(display_summary = False)
        self.model.load_weights(directory_path + '/model_weights')
        self._save_path = directory_path

        self._y_test_pred = self.model.predict(self.X_test, verbose=0)
        self._y_test_pred = self._y_test_pred.reshape(len(self.y_test))

        
        
      
    
    def make_plot_results_test_R2(self, 
                                  figsize=(8, 8), xfontsize = 37, yfontsize = 37, title_fontsize = 42,
                                  ticks_fontsize = 30, r2_fontsize = 28, r2_loc = (0.06, 0.95),
                                  legend_fontsize = 23, legend_loc = (0.70,0.02), start_day = 7, end_day = 17,
                                  xticks = [0,10,20,30], yticks = [0,10,20,30],
                                  save_fig = False
                                 ):

        """
        Makes a scatterplot that illustrates the differences between the true values and the corresponding predicted values for each point in the test set. The scatterplots are further divided to distinguish between the daylight (red) and nighttime (lightblue) periods.

        Parameters:
            figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (8, 8).
            xfontsize (int, optional): Font size for x-axis labels. Defaults to 37.
            yfontsize (int, optional): Font size for y-axis labels. Defaults to 37.
            title_fontsize (int, optional): Font size for the title. Defaults to 42.
            ticks_fontsize (int, optional): Font size for the axis ticks. Defaults to 30.
            r2_fontsize (int, optional): Font size for the R-squared value. Defaults to 28.
            r2_loc (tuple, optional): Location (x, y) of the R-squared value in the plot. Defaults to (0.06, 0.95).
            legend_fontsize (int, optional): Font size for the legend. Defaults to 23.
            legend_loc (tuple, optional): Location (x, y) of the legend in the plot. Defaults to (0.70, 0.02).
            start_day (int, optional): Hour that define the start of the daylight period. Defaults to 7.
            end_day (int, optional): Hour that define the end of the daylight period. Defaults to 17.
            save_fig (bool, optional): Whether to save the figure. Defaults False.
    
        Returns:
            fig,ax
        """                             
                                     
        if self.target_variable == 'T_target':
            xlabel = 'Temperature °C'

        if self.target_variable == 'H_target':
            xlabel = 'Relative humidity %'

            
        fig, ax = plot_results_test_R2(x = self.y_test, y = self._y_test_pred, hour = self.df_test.Hour.values,
                                      title = self.model_name, xlabel = xlabel,
                                      figsize=figsize, xfontsize = xfontsize, yfontsize = yfontsize,
                                      title_fontsize = title_fontsize, ticks_fontsize = ticks_fontsize,
                                      r2_fontsize = r2_fontsize, r2_loc = r2_loc,
                                      legend_fontsize = legend_fontsize, legend_loc = legend_loc,
                                      start_day = start_day, end_day = end_day,
                                      xticks = xticks, yticks = yticks
                                      )
                                     
        if save_fig:
            fig.savefig(self._save_path + '/figures/figure_results_test_R2.png', dpi=150)
        return fig,ax


    def make_plot_distribution_test_mae(self,
                                        thr=-1,stat='density',
                                        figsize=(8, 6), xfontsize = 35, yfontsize = 35, title_fontsize = 38,
                                        ticks_fontsize = 25, mae_fontsize = 25, mae_loc = (0.70, 0.68),
                                        legend_fontsize = 25, start_day = 7, end_day = 17,
                                        save_fig = False
                                       ):
        """
        Plots the distribution of mean absolute errors within the test set, categorized by the daylight (red) and nighttime (lightblue) periods. The histograms have been normalized to ensure that the area under each histogram is equal to 1. The R2 value computed across the entire test set is represented in the upper left corner.

        Parameters:
            thr (float, optional): Threshold value to set the min mae value to plot. Defaults to -1 (plot all mae values).
            stat (str, optional): Type of statistic to display. 'density' or 'count'. Defaults to 'density'.
            figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (8, 6).
            xfontsize (int, optional): Font size for x-axis labels. Defaults to 28.
            yfontsize (int, optional): Font size for y-axis labels. Defaults to 28.
            title_fontsize (int, optional): Font size for the title. Defaults to 35.
            ticks_fontsize (int, optional): Font size for the axis ticks. Defaults to 18.
            mae_fontsize (int, optional): Font size for the MAE value. Defaults to 20.
            mae_loc (tuple, optional): Location (x, y) of the MAE value in the plot. Defaults to (0.78, 0.75).
            legend_fontsize (int, optional): Font size for the legend. Defaults to 18.
            start_day (int, optional): Hour that define the start of the daylight period. Defaults to 7.
            end_day (int, optional): Hour that define the end of the daylight period. Defaults to 17.
            save_fig (bool, optional): Whether to save the figure. Defaults False.
    
        Returns:
            fig,ax
        """
        if self.target_variable == 'T_target':
            xlabel = "Absolute error [°C]"

        if self.target_variable == 'H_target':
            xlabel = "Absolute error [%]"
            
        fig, ax = plot_distribution_test_mae(x = self.y_test, y = self._y_test_pred, hour = self.df_test.Hour.values,
                                             title = self.model_name, xlabel = xlabel, thr = thr, stat = stat,
                                             figsize=figsize, xfontsize = xfontsize, yfontsize = yfontsize,
                                             title_fontsize = title_fontsize, ticks_fontsize = ticks_fontsize,
                                             mae_fontsize = mae_fontsize, mae_loc = mae_loc,
                                             legend_fontsize = legend_fontsize,
                                             start_day = start_day, end_day = end_day
                                            )

        if save_fig:
            fig.savefig(self._save_path + '/figures/figure_distribution_test_mae.png', dpi=150)                                   
        return fig, ax



    def make_plot_inputs_importance(self,n_simul,
                                    figsize = (8,6), xfontsize = 33, yfontsize = 33,
                                    title_fontsize = 40,  ticks_fontsize = 25, title = None, save_fig = False
                                   ):
        """
        Plots the importance of input variables in a feed-forward neural network.
        
        Parameters:
            n_simul (int): Number of simulations to perform.
            model_name (str): Name of the model.
            figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (12, 7).
            xfontsize (int, optional): Font size for x-axis labels. Defaults to 30.
            yfontsize (int, optional): Font size for y-axis labels. Defaults to 30.
            title_fontsize (int, optional): Font size for the title. Defaults to 40.
            ticks_fontsize (int, optional): Font size for the axis ticks. Defaults to 20.
            save_fig (bool, optional): Whether to save the figure. Defaults False.
        
        Returns:
            fig,ax
        """
        if not title:
            model_name = self.model_name
        else:
            model_name = title

        fig,ax = plot_inputs_importance(model = self.model, df_test = self.df_test,
                                        X_test = self.X_test, y_test = self.y_test, input_variables = self.input_variables,
                                        n_simul = n_simul, model_name = model_name,
                                        figsize = figsize,
                                        xfontsize = xfontsize, yfontsize = yfontsize,
                                        title_fontsize = title_fontsize, ticks_fontsize = ticks_fontsize
                                       )

        if save_fig:
            fig.savefig(self._save_path + '/figures/figure_inputs_importance.png', dpi=150)                                   
        return fig, ax
                                       

    def make_plot_cross_period_prediction(self, df_filename_cross_period, cross_period_name,
                                          figsize_r2=(8, 8), xfontsize_r2 = 27, yfontsize_r2 = 27, title_fontsize_r2 = 30,
                                          r2_ticks_fontsize = 20,r2_fontsize = 20, r2_loc = (0.03, 0.95), r2_legend_fontsize = 16, 
                                          r2_legend_loc =(0.80,0.02), 
                                          figsize_mae=(8, 6), xfontsize_mae = 25, yfontsize_mae = 25, title_fontsize_mae = 30,
                                          mae_ticks_fontsize = 20, mae_fontsize = 20, mae_loc = (0.77, 0.75),mae_legend_fontsize = 18,
                                          start_day = 7, end_day = 17, save_fig = False
                                         ):
        """
        Plots Cross-Period Prediction. The plots include the R2 scores and mean absolute errors for the cross-period predictions.
        
        Parameters:
            df_filename_cross_period (str): The filename of the dataset containing cross-period data.
            cross_period_name (str): The name of the cross-period.
            figsize_r2 (tuple, optional): The figure size (width, height) for the R2 plot. Default: (8, 8).
            xfontsize_r2 (int, optional): The font size for the x-axis labels in the R2 plot. Default: 27.
            yfontsize_r2 (int, optional): The font size for the y-axis labels in the R2 plot. Default: 27.
            title_fontsize_r2 (int, optional): The font size for the title of the R2 plot. Default: 30.
            r2_ticks_fontsize (int, optional): The font size for the tick labels in the R2 plot. Default: 20.
            r2_fontsize (int, optional): The font size for the R-squared value text in the R2 plot. Default: 20.
            r2_loc (tuple, optional): The location (x, y) of the R-squared value text in the R2 plot. Default: (0.03, 0.95).
            r2_legend_fontsize (int, optional): The font size for the legend in the R2 plot. Default: 16.
            r2_legend_loc (tuple, optional): The location (x, y) of the legend in the R2 plot. Default: (0.80, 0.02).
            figsize_mae (tuple, optional): The figure size (width, height) for the MAE plot. Default: (8, 6).
            xfontsize_mae (int, optional): The font size for the x-axis labels in the MAE plot. Default: 25.
            yfontsize_mae (int, optional): The font size for the y-axis labels in the MAE plot. Default: 25.
            title_fontsize_mae (int, optional): The font size for the title of the MAE plot. Default: 30.
            mae_ticks_fontsize (int, optional): The font size for the tick labels in the MAE plot. Default: 20.
            mae_fontsize (int, optional): Font size for the MAE value. Defaults to 20.
            mae_loc (tuple, optional, optional): Location (x, y) of the MAE value in the plot. Defaults to (0.77, 0.75).
            mae_legend_fontsize (int, optional): Font size for the legend in the MAE plot. Defaults to 18.
            start_day (int, optional): Hour that define the start of the daylight period. Defaults to 7.
            end_day (int, optional): Hour that define the end of the daylight period. Defaults to 17.
            save_fig (bool, optional): Whether to save the figure. Defaults False.
            
        Returns:
         fig_r2, ax_r2, fig_mae, ax_mae
    
        """

        fig_r2, ax_r2, fig_mae, ax_mae  = plot_cross_period_prediction(
                                             df_filename_cross_period = df_filename_cross_period,
                                             model = self.model, 
                                             train_data_rescale_params = self.train_data_rescale_params,
                                             input_variables = self.input_variables,
                                             target_variable = self.target_variable, 
                                             period_name_model = self.model_name,
                                             cross_period_name = cross_period_name,
                                             figsize_r2=figsize_r2, xfontsize_r2 = xfontsize_r2, yfontsize_r2 = yfontsize_r2,
                                             title_fontsize_r2 = title_fontsize_r2, r2_ticks_fontsize = r2_ticks_fontsize,
                                             r2_fontsize = r2_fontsize, r2_loc = r2_loc, 
                                             r2_legend_fontsize = r2_legend_fontsize, r2_legend_loc = r2_legend_loc, 
                                             figsize_mae=figsize_mae, xfontsize_mae = xfontsize_mae, yfontsize_mae = yfontsize_mae,
                                             title_fontsize_mae = title_fontsize_mae, mae_ticks_fontsize = mae_ticks_fontsize,
                                             mae_fontsize = mae_fontsize, mae_loc = mae_loc, mae_legend_fontsize = mae_legend_fontsize,
                                             start_day = start_day, end_day = end_day
                                                )

        if save_fig:
            fig_r2.savefig(self._save_path + '/figures/figure_cross_period_results_R2.png', dpi=150) 
            fig_mae.savefig(self._save_path + '/figures/figure_cross_period_distribution_mae.png', dpi=150) 
            
        return fig_r2, ax_r2, fig_mae, ax_mae


    def make_plot_microclimate_prediction_over_study_area(self, df_climate, period, lat, long,
                                                          dsm, resolution, slope, aspect,ha, svf,study_area_coordinates,
                                                          figsize=(8,6), aspect_heatmap = 1, shrink = 0.7,
                                                          tick_params_labelsize = 20, ylabel_fontsize = 35,
                                                          title_fontsize = 30, save_figure=False,
                                                         ):
        """
        Showcases the predicted temperature variations at a specific time instance over the entire study area.
        
        Parameters:
            df_climate (DataFrame): The dataset containing climate data.
            period (str): The period of interest.
            lat (float): The latitude values of the study area.
            long (float): The longitude values of the study area.
            dsm (numpy.ndarray): The digital surface model data of the study area.
            resolution (float): The resolution of the study area data.
            slope (numpy.ndarray): The slope data of the study area.
            aspect (numpy.ndarray): The aspect data of the study area.
            ha (numpy.ndarray): The horizontal aspect data of the study area.
            svf (numpy.ndarray): The sky view factor data of the study area.
            study_area_coordinates (list): The coordinates of the study area corners.
            figsize (tuple, optional): The figure size (width, height) for the plot. Default: (8, 6).
            aspect_heatmap (float, optional): The aspect ratio for the heatmap plot. Default: 1.
            shrink (float, optional): The size of the colorbar relative to the heatmap plot. Default: 0.7.
            tick_params_labelsize (int, optional): The font size for the tick labels. Default: 20.
            ylabel_fontsize (int, optional): The font size for the y-axis label. Default: 35.
            title_fontsize (int, optional): The font size for the plot title. Default: 30.
            save_figure (bool, optional): Whether to save the figure. Default: False.
            figure_filename (str, optional): The filename to save the figure. Required if save_figure is True.
            save_figure (bool, optional): Whether to save the figure. Defaults False.
        
        Returns:
            fig, ax
        """ 
                                                             
        if save_figure:
            if self.target_variable == 'T_target': 
                var_save_filename = 'temperature'
            if self.target_variable == 'H_target': 
                var_save_filename = 'humidity'
            figure_filename=self._save_path +'/figures/{}_{}_spacemap_{}_{}_{:02d}_{:02d}.png'.format(self.model_type_name,var_save_filename,period[0], period[1], period[2], period[3])
        else:
            figure_filename=None

                                                             
        fig, ax = plot_microclimate_prediction_over_study_area(model = self.model, target_variable = self.target_variable,
                                                               train_data_rescale_params = self.train_data_rescale_params,
                                                               df_climate = df_climate, period = period,
                                                               lat = lat, long = long, dsm = dsm, resolution = resolution,
                                                               slope = slope, aspect = aspect, ha = ha, svf = svf,
                                                               study_area_coordinates = study_area_coordinates,
                                                               figsize=figsize, aspect_heatmap = aspect_heatmap, shrink = shrink,
                                                               tick_params_labelsize = tick_params_labelsize,
                                                               ylabel_fontsize = ylabel_fontsize, title_fontsize = title_fontsize,
                                                               save_figure = save_figure, figure_filename = figure_filename)

        return fig, ax



    
                                                             
    def compute_microclimate_prediction_over_study_area(self, df_climate, period, lat, long,
                                                          dsm, resolution, slope, aspect,ha, svf,study_area_coordinates
                                                         ):
        """
        Computes the predicted temperature variations at a specific time instance over the entire study area.
        
        Parameters:
            df_climate (DataFrame): The dataset containing climate data.
            period (str): The period of interest.
            lat (float): The latitude values of the study area.
            long (float): The longitude values of the study area.
            dsm (numpy.ndarray): The digital surface model data of the study area.
            resolution (float): The resolution of the study area data.
            slope (numpy.ndarray): The slope data of the study area.
            aspect (numpy.ndarray): The aspect data of the study area.
            ha (numpy.ndarray): The horizontal aspect data of the study area.
            svf (numpy.ndarray): The sky view factor data of the study area.
            study_area_coordinates (list): The coordinates of the study area corners.
    
        Returns:
            Microclimate map
        """ 
                                                             
        

                                                             
        year = period[0]
        month = period[1]
        day = period[2]
        hour = period[3]
        
        df_climate_micro = df_climate[(df_climate.Year == year)&(df_climate.Day == day) & (df_climate.Hour == hour)&(df_climate.Month == month)]
        assert len(df_climate_micro) == 1 
        
        microclima_map = make_microclimate_prediction_over_study_area(
                            model = self.model,
                            train_data_rescale_params = self.train_data_rescale_params,
                            df_climate_micro = df_climate_micro, year = year, month = month, day = day, hour = hour,
                            lat = lat, long = long, dsm = dsm, resolution = resolution,
                            slope = slope, aspect = aspect, ha = ha, svf = svf, study_area_coordinates = study_area_coordinates)

        return microclima_map


                                                             


    def compute_mae_temporal_evolution(self,):

        mae_sensors = []

        for sensor_id in self.df_test.Sensor_id.unique():
            dfs = self.df_test[self.df_test.Sensor_id == sensor_id]

            Xs_test, ys_test = get_data_from_df(dfs,
                                                input_variables = self.input_variables,
                                                target_variable = self.target_variable) 
    
            Xs_test = ((Xs_test - self.train_data_rescale_params['mean']) / self.train_data_rescale_params['std'])
            preds = self.model.predict(Xs_test, verbose=0)
            preds = preds.reshape(len(ys_test))
            maes = np.abs(ys_test-preds)
            mae_sensors.append(maes)

        mae_sensors = np.mean(mae_sensors, axis = 0)

        return mae_sensors


    

    
        
            






            
    

        
