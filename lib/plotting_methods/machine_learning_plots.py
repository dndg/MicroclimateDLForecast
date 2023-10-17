import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import copy
import sys
import matplotlib as mpl

USER_PATH = '/Users/marcozanchi/Work/Grade/'
LIB_PATH = USER_PATH + 'microclima/lib/'

from heatmpas_plotting_methods import make_heatmap

sys.path.append(LIB_PATH + 'machine_learning')
from elaborate_data_utility import get_data_from_df

sys.path.append(LIB_PATH + 'microclima_generation')
from predict_microclimate_over_study_area import make_microclimate_prediction_over_study_area
 




def plot_history(history_df, color_train, color_val, title,
                 figsize = (8, 6), ms = 7, ylabel = "Loss function", xfontsize = 25, yfontsize=25, title_fontsize = 30):
    """
    Plots the training and validation history of a model.

    Parameters:
        history_df (pandas.DataFrame): DataFrame containing the training history. 
        color_train (str): Color for the training loss line.
        color_val (str): Color for the validation loss line.
        title (str): Title of the plot.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (8, 6).
        ms (int, optional): Marker size for data points. Defaults to 7.
        ylabel (str, optional): Label for the y-axis. Defaults to "Loss function".
        xfontsize (int, optional): Font size for x-axis labels. Defaults to 25.
        yfontsize (int, optional): Font size for y-axis labels. Defaults to 25.
        title_fontsize (int, optional): Font size for the title. Defaults to 30.

    Returns:
        fig,ax
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(history_df.loss, marker="o", ms=ms, label="Train set", color=color_train)
    ax.plot(history_df.val_loss, marker="o", ms=ms, label="Validation set", color=color_val)
    ax.legend(frameon=False, fontsize=18)
    ax.set_xlabel("Epoch",fontsize = xfontsize)
    ax.set_ylabel(ylabel,fontsize = yfontsize)
    ax.set_title(title, fontsize = title_fontsize)                  

    ax.tick_params(labelsize = 18)                 
    fig.tight_layout()
    return fig, ax




def plot_results_test_R2(x, y, hour, title, xlabel,
                         figsize=(8, 8), xfontsize = 30, yfontsize = 30, title_fontsize = 38,
                         ticks_fontsize = 20, r2_fontsize = 20, r2_loc = (0.06, 0.95),
                         legend_fontsize = 16, legend_loc = (0.77,0.02), start_day = 7, end_day = 17,
                         xticks = [0,10,20,30], yticks = [0,10,20,30]
                        ):
    """
    Makes a scatterplot that illustrates the differences between the true values and the corresponding predicted values for each point in the test set. The scatterplots are further divided to distinguish between the daylight (red) and nighttime (lightblue) periods.

    Parameters:
        x (numpy.ndarray): Array of true values.
        y (numpy.ndarray): Array of predicted values.
        hour (numpy.ndarray): Array of hour values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (8, 8).
        xfontsize (int, optional): Font size for x-axis labels. Defaults to 30.
        yfontsize (int, optional): Font size for y-axis labels. Defaults to 30.
        title_fontsize (int, optional): Font size for the title. Defaults to 38.
        ticks_fontsize (int, optional): Font size for the axis ticks. Defaults to 20.
        r2_fontsize (int, optional): Font size for the R-squared value. Defaults to 20.
        r2_loc (tuple, optional): Location (x, y) of the R-squared value in the plot. Defaults to (0.06, 0.95).
        legend_fontsize (int, optional): Font size for the legend. Defaults to 16.
        legend_loc (tuple, optional): Location (x, y) of the legend in the plot. Defaults to (0.77, 0.02).
        start_day (int, optional): Hour that define the start of the daylight period. Defaults to 7.
        end_day (int, optional): Hour that define the end of the daylight period. Defaults to 17.

    Returns:
        fig,ax
    """
                            
    R2 = linregress(x, y).rvalue ** 2

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    hour = np.where((hour>=start_day)*(hour<=end_day),'Day','Night')
    sns.scatterplot(x = x, y = y, hue = hour,palette=['lightblue','tomato'],ax=ax)

    ax.set_xlabel(xlabel, fontsize = xfontsize)
    ax.set_ylabel("Model prediction", fontsize = yfontsize)
    ax.set_title(title, fontsize = title_fontsize)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize = ticks_fontsize)  

    ax.set_aspect(1)
    ax.text(r2_loc[0], r2_loc[1], f"$R^2$ = {R2:.2f}", fontsize=r2_fontsize, transform=ax.transAxes, va="top", ha="left")
    ax.plot([np.min(x), np.max(x)], [np.min(y), np.max(y)], ls="dashed", c="0.2")
    ax.legend(loc= legend_loc, fontsize = legend_fontsize)
    fig.tight_layout()
    return fig, ax




def plot_distribution_test_mae(x, y, hour, title, xlabel, thr=-1,stat='density',
                               figsize=(8, 6), xfontsize = 28, yfontsize = 28, title_fontsize = 35,
                               ticks_fontsize = 18, mae_fontsize = 20, mae_loc = (0.78, 0.75),
                               legend_fontsize = 18, start_day = 7, end_day = 17 ):
    """
    Plots the distribution of mean absolute errors within the test set, categorized by the daylight (red) and nighttime (lightblue) periods. The histograms have been normalized to ensure that the area under each histogram is equal to 1. The R2 value computed across the entire test set is represented in the upper left corner.

    Parameters:
        x (numpy.ndarray): Array of true values.
        y (numpy.ndarray): Array of predicted values.
        hour (numpy.ndarray): Array of hour values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
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

    Returns:
        fig,ax
    """

                                    
    mpl.rcParams['legend.fontsize'] = legend_fontsize 
                                   
    MAE = np.mean(np.abs(x-y))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    hour = np.where((hour>=start_day)*(hour<=end_day),'Day','Night')
                                   
    mae = np.abs(x-y)
    
    mae_thr = mae[mae>thr]
    hour_thr = hour[mae>thr]
    gfg = sns.histplot(x = mae_thr,stat=stat,element="step", ax = ax,
                       hue = hour_thr,palette=['lightblue','tomato'],
                       multiple='stack') #label = 'Mae prediction'

    ax.set_xlabel(xlabel, fontsize = xfontsize)
    ax.set_ylabel("Density", fontsize = yfontsize)
    ax.set_title(title, fontsize = title_fontsize)
    ax.tick_params(labelsize = ticks_fontsize)                            

    ax.text(mae_loc[0], mae_loc[1], f"$MAE$ = {MAE:.2f}", fontsize=mae_fontsize, transform=ax.transAxes, va="top", ha="left")
                                   
    fig.tight_layout()
                                   
    return fig, ax      


                              

def plot_inputs_importance(model, df_test, X_test, y_test, input_variables, n_simul, model_name,
                           figsize = (12,7), xfontsize = 30, yfontsize = 30, title_fontsize = 40,  ticks_fontsize = 20
                          ):
    """
    Plots the importance of input variables in a feed-forward neural network.

    Parameters:
        model (object): Trained model object.
        df_test (pandas.DataFrame): DataFrame of test data.
        X_test (numpy.ndarray): Array of input features for the test data.
        y_test (numpy.ndarray): Array of target values for the test data.
        input_variables (list): List of input variable names.
        n_simul (int): Number of simulations to perform.
        model_name (str): Name of the model.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (12, 7).
        xfontsize (int, optional): Font size for x-axis labels. Defaults to 30.
        yfontsize (int, optional): Font size for y-axis labels. Defaults to 30.
        title_fontsize (int, optional): Font size for the title. Defaults to 40.
        ticks_fontsize (int, optional): Font size for the axis ticks. Defaults to 20.

    Returns:
        fig,ax
    """
    
    y_test_pred = model.predict(X_test, verbose = False)
    y_test_pred = y_test_pred.reshape(len(y_test))
    R2_true = linregress(y_test, y_test_pred).rvalue ** 2
    Mae_true = np.mean(np.abs( y_test - y_test_pred))

    metadata_listdict = []

    rename_dict = {
        'Short-wave radiation':'Swr',
        'Long-wave radiation':'Lwr',
        'Wind':'Ws',
        'T_ref':'Tr',
        'H_ref':'Hr',
        'Precipitation':'Prc',
        'Pressure':'P',
        'Cloud-cover':'Cc',
        'Specific-humidity':'Hs'
                  }                              
                              

    for input_idx in range(len(input_variables)):
        for _ in range(n_simul):
            X_test_random = copy.deepcopy(X_test)
            Min = np.min(X_test_random[:,input_idx])
            Max = np.max(X_test_random[:,input_idx])
            X_test_random[:,input_idx] = np.random.uniform(Min,Max,len(X_test_random[:,input_idx]))

            y_test_pred_rand = model.predict(X_test_random, verbose = False)
            y_test_pred_rand = y_test_pred_rand.reshape(len(y_test))

            metadata_dict = {
                'Variable':rename_dict[input_variables[input_idx]],
                'R2': R2_true - linregress(y_test, y_test_pred_rand).rvalue ** 2 ,
                'Mae': np.mean(np.abs(y_test - y_test_pred_rand)) - Mae_true,
            }
            metadata_listdict.append(metadata_dict)

    df = pd.DataFrame(metadata_listdict)
    fig, axs = plt.subplots(1,2, figsize = figsize, sharey = True)

    ax = axs[0]
    sns.barplot(data = df, y = 'Variable', x = 'R2',
                ax = ax,color = 'orangered')
    ax.set_xlabel(r'$R^2-R^2_{rand}$', fontsize = xfontsize)
    ax.set_ylabel('Input variable', fontsize = yfontsize)
    ax.tick_params(labelsize = ticks_fontsize) 

                                  
    ax = axs[1]
    sns.barplot(data = df, y = 'Variable', x = 'Mae',
                ax = ax,color = 'slateblue')
    ax.set_xlabel(r'$MAE_{rand}-MAE$', fontsize = xfontsize)
    ax.set_ylabel('')
    ax.tick_params(labelsize = ticks_fontsize) 
                              
                              
    fig.suptitle(model_name , fontsize = title_fontsize)

    fig.tight_layout()

    return fig, ax



def plot_cross_period_prediction(df_filename_cross_period, model, train_data_rescale_params, input_variables, target_variable,
                                 period_name_model, cross_period_name,
                                 figsize_r2=(8, 8), xfontsize_r2 = 30, yfontsize_r2 = 30, title_fontsize_r2 = 34, r2_ticks_fontsize =20,
                                 r2_fontsize = 20, r2_loc = (0.03, 0.95), r2_legend_fontsize = 16, r2_legend_loc = (0.80,0.02), 
                                 figsize_mae=(8, 6), xfontsize_mae = 28, yfontsize_mae = 28, title_fontsize_mae = 34,
                                 mae_ticks_fontsize = 20, mae_fontsize = 20, mae_loc = (0.77, 0.75),mae_legend_fontsize = 18,
                                 start_day = 7, end_day = 17
                                ):
    """
    Plots Cross-Period Prediction. The plots include the R2 scores and mean absolute errors for the cross-period predictions.
    
    Parameters:
        df_filename_cross_period (str): The filename of the dataset containing cross-period data.
        model (object): The trained machine learning model used for making predictions.
        train_data_rescale_params (dict): A dictionary containing the rescaling parameters used for training the model.
        input_variables (list): A list of strings representing the names of the input variables used for prediction.
        target_variable (str): The name of the target variable to be predicted.
        period_name_model (str): The name of the period in the model.
        cross_period_name (str): The name of the cross-period.
        figsize_r2 (tuple, optional): The figure size (width, height) for the R2 plot. Default: (8, 8).
        xfontsize_r2 (int, optional): The font size for the x-axis labels in the R2 plot. Default: 30.
        yfontsize_r2 (int, optional): The font size for the y-axis labels in the R2 plot. Default: 30.
        title_fontsize_r2 (int, optional): The font size for the title of the R2 plot. Default: 34.
        r2_ticks_fontsize (int, optional): The font size for the tick labels in the R2 plot. Default: 20.
        r2_fontsize (int, optional): The font size for the R-squared value text in the R2 plot. Default: 20.
        r2_loc (tuple, optional): The location (x, y) of the R-squared value text in the R2 plot. Default: (0.03, 0.95).
        r2_legend_fontsize (int, optional): The font size for the legend in the R2 plot. Default: 16.
        r2_legend_loc (tuple, optional): The location (x, y) of the legend in the R2 plot. Default: (0.80, 0.02).
        figsize_mae (tuple, optional): The figure size (width, height) for the MAE plot. Default: (8, 6).
        xfontsize_mae (int, optional): The font size for the x-axis labels in the MAE plot. Default: 28.
        yfontsize_mae (int, optional): The font size for the y-axis labels in the MAE plot. Default: 28.
        title_fontsize_mae (int, optional): The font size for the title of the MAE plot. Default: 34.
        mae_ticks_fontsize (int, optional): The font size for the tick labels in the MAE plot. Default: 20.
        mae_fontsize (int, optional): Font size for the MAE value. Defaults to 20.
        mae_loc (tuple, optional, optional): Location (x, y) of the MAE value in the plot. Defaults to (0.78, 0.75).
        mae_legend_fontsize (int, optional): Font size for the legend in the MAE plot. Defaults to 18.
        start_day (int, optional): Hour that define the start of the daylight period. Defaults to 7.
        end_day (int, optional): Hour that define the end of the daylight period. Defaults to 17.
        
    Returns:
     fig_r2, ax_r2, fig_mae, ax_mae

    """
                                     

    df_cross_period = pd.read_pickle(df_filename_cross_period)
    X_cross_period, y_cross_period = get_data_from_df(df_cross_period,
                                                      input_variables = input_variables, target_variable = target_variable)
    m = train_data_rescale_params['mean']
    s = train_data_rescale_params['std']
    
    X_cross_period =  ((X_cross_period - m) / s)
    y_cross_period_pred = model.predict(X_cross_period, verbose=0)
    y_cross_period_pred = y_cross_period_pred.reshape(len(y_cross_period))

    if target_variable == 'T_target':
        xlabel_R2 = 'Temperature °C'
        xlabel_mae = "Absolute error [°C]"
        
    elif target_variable == 'H_target':
        xlabel_R2 = 'Relative humidity %'
        xlabel_mae = "Absolute error [%]"
                                     
    
    fig_r2,ax_r2 = plot_results_test_R2(y_cross_period, y_cross_period_pred, df_cross_period.Hour.values,
                                  title='{} cross period on {}'.format(period_name_model, cross_period_name),
                                  xlabel = xlabel_R2,
                                  figsize = figsize_r2, xfontsize = xfontsize_r2, yfontsize = yfontsize_r2,
                                  title_fontsize = title_fontsize_r2, ticks_fontsize = r2_ticks_fontsize,
                                  r2_fontsize = r2_fontsize, r2_loc = r2_loc,
                                  legend_fontsize = r2_legend_fontsize, legend_loc = r2_legend_loc,
                                  start_day = start_day, end_day = end_day,
                                 )
                                     
    fig_mae,ax_mae = plot_distribution_test_mae(y_cross_period, y_cross_period_pred, df_cross_period.Hour.values,
                                        title='{} cross period on {}'.format(period_name_model, cross_period_name),
                                        xlabel = xlabel_mae,
                                        thr=-1,stat='density',
                                        figsize = figsize_mae, xfontsize = xfontsize_mae, yfontsize = yfontsize_mae,
                                        title_fontsize = title_fontsize_mae, ticks_fontsize = mae_ticks_fontsize,
                                        mae_fontsize = mae_fontsize, mae_loc = mae_loc, legend_fontsize = mae_legend_fontsize,
                                        start_day = start_day, end_day = end_day,
                                       )                              

    return fig_r2, ax_r2, fig_mae, ax_mae




def plot_microclimate_prediction_over_study_area(model, target_variable, train_data_rescale_params,
                                                 df_climate, period,
                                                 lat, long, dsm, resolution, slope, aspect,ha, svf,study_area_coordinates,
                                                 figsize=(8,6), aspect_heatmap = 1, shrink = 0.7,
                                                 tick_params_labelsize = 20, ylabel_fontsize = 35, title_fontsize = 30,
                                                 save_figure=False, figure_filename=None
                                                ):
                                                    
    """
    Showcases the predicted temperature variations at a specific time instance over the entire study area.
    
    Parameters:
        model (object): The trained machine learning model used for making predictions.
        target_variable (str): The name of the target variable to be predicted.
        train_data_rescale_params (dict): A dictionary containing the rescaling parameters used for training the model.
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
    
    Returns:
        fig, ax
    """                                                
    year = period[0]
    month = period[1]
    day = period[2]
    hour = period[3]
    
    df_climate_micro = df_climate[(df_climate.Year == year)&(df_climate.Day == day) & (df_climate.Hour == hour)&(df_climate.Month == month)]
    assert len(df_climate_micro) == 1 

    microclima_map = make_microclimate_prediction_over_study_area(
                        model = model, train_data_rescale_params = train_data_rescale_params,
                        df_climate_micro = df_climate_micro, year = year, month = month, day = day, hour = hour,
                        lat = lat, long = long, dsm = dsm, resolution = resolution,
                        slope = slope, aspect = aspect, ha = ha, svf = svf, study_area_coordinates = study_area_coordinates)


    if target_variable == 'T_target':           
        cmap = 'inferno'
        cbar_label = 'Temperature [°C]'
        var_title = 'Temperature'
        
            
    if target_variable == 'H_target':          
        cmap = 'cool'
        cbar_label = 'Relative humidity %'
        var_title = 'Relative humidity'
        

                                                     
    fig, ax = make_heatmap(microclima_map, cmap = cmap, vmin = np.min(microclima_map)-1, vmax = np.max(microclima_map)+1,
                          cbar_label = cbar_label,
                          title = '{} {:02d}.{:02d}.{} - h{:02d}'.format(var_title, day, month, year, hour),
                          figsize=figsize, aspect = aspect_heatmap, shrink = shrink,
                          tick_params_labelsize = tick_params_labelsize, ylabel_fontsize = ylabel_fontsize,
                          title_fontsize = title_fontsize, save_figure=save_figure, figure_filename=figure_filename
                          )

    return fig,ax
                                                    











                                    