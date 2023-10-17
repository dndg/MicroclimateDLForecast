import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, LSTM




def get_model(layers, learning_rate, activation, loss, metrics,input_shape):
    """
    Create a neural network model with the specified architecture and settings.

    Parameters:
        layers (list): A list of integers representing the number of neurons in each hidden layer.
        learning_rate (float): The learning rate for the Adam optimizer.
        activation (str): The activation function to be used in the hidden layers.
        loss (str): The loss function to be used during training.
        metrics (list): A list of metrics to be evaluated during training.
        input_shape (tuple): The shape of the input data.

    Returns:
        keras.Model: A compiled Keras model with the specified architecture and settings.


    """

    input_model = Input(shape=input_shape)
    
    fc = Dense(layers[0], activation = activation)(input_model)
    
    for layer in layers[1:]:
        fc = Dense(layer, activation = activation)(fc)

    output = Dense(1)(fc)

    model = Model(inputs = input_model, outputs = output)
    model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                metrics = metrics)
    return model





def create_model(max_n_neurons, min_n_neurons, n_layers, learning_rate, activation, loss, metrics, input_shape):
    """
    Create a neural network model with a flexible number of layers and neurons.

    Parameters:
        max_n_neurons (int): The maximum number of neurons allowed in each layer.
        min_n_neurons (int): The minimum number of neurons allowed in each layer.
        n_layers (int): The number of hidden layers in the model.
        learning_rate (float): The learning rate for the optimizer.
        activation (str): The activation function to be used in the hidden layers.
        loss (str): The loss function to be used during training.
        metrics (list): A list of metrics to be evaluated during training.
        input_shape (tuple): The shape of the input data.

    Returns:
        keras.Model: A compiled Keras model with the flexible number of layers and neurons.
    """

    neurons = max_n_neurons
    layers = []
    while neurons >= min_n_neurons:
        layers.append(neurons)
        neurons = int(neurons/2)    
    assert len(layers) == n_layers
    assert np.max(layers) == max_n_neurons
    assert np.min(layers) == min_n_neurons

    model = get_model(layers = layers, learning_rate = learning_rate, activation = activation,
                      loss = loss, metrics = metrics, input_shape = input_shape)
    return model



def train_model(model, X_train, y_train, X_validation, y_validation, epochs, patience,
                metric_to_monitor = 'val_loss',restore_best_weights = True, verbose = 1):
    """
    Train a neural network model with the given training and validation data.

    Parameters:
        model (keras.Model): The compiled Keras model to be trained.
        X_train (numpy.ndarray): The input training data.
        y_train (numpy.ndarray): The target training data.
        X_validation (numpy.ndarray): The input validation data.
        y_validation (numpy.ndarray): The target validation data.
        epochs (int): The number of training epochs.
        patience (int): The number of epochs with no improvement after which training will be stopped if `restore_best_weights` is set to True.
        metric_to_monitor (str, optional): The metric to be monitored for early stopping. Defaults to 'val_loss'.
        restore_best_weights (bool, optional): Whether to restore the weights of the best model during training. Defaults to True.
        verbose (int, optional): Verbosity mode (0 = silent, 1 = progress bar). Defaults to 1.
    Returns:
        history_df (pd.DataFrame): Pandas dataframe containing the loss values for each epochs for the training ad validation.
        model (keras.Model): The trained Keras model.
    """

    history = model.fit(X_train, y_train,
                        validation_data = (X_validation, y_validation),
                        epochs= epochs,
                        verbose = verbose,
                        callbacks = tf.keras.callbacks.EarlyStopping(patience = patience,
                                                                     monitor = metric_to_monitor,
                                                                     restore_best_weights = restore_best_weights
                                                                    )
                       )
    
    
    history_df = pd.DataFrame(history.history)    
    
    return model, history_df