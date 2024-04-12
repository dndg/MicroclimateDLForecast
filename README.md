# Project description
This repository contains the pyhton code and the jupyter notebooks which reproduce all the results and figures presented in the paper: Zanchi, M. et al Harnessing deep learning to forecast local microclimate using global climate data. Sci Rep 13, 21062 (2023). https://doi.org/10.1038/s41598-023-48028-1

# notebooks
The 'notebooks' directory contains a collection of notebooks that provide detailed explanations of the research procedures, theoretical models employed, and the presentation of results. The following notebook is included:
- notebook_physical_model: This notebook elucidates the physical model implemented in the current research, which enables the downscaling of environmental physical variables at the meter scale.
- notebook_feed_forward_neural_network_temperature_prediction: This notebook contains all the results concerning the training, validation, tetsing and transfer learning of the feed forward neural network trained for predicting the temperature.
- notebook_feed_forward_neural_network_humidity_prediction: This notebook contains all the results concerning the training, validation, tetsing and transfer learning of the feed forward neural network trained for predicting the relative humidity.

# lib
The 'lib' directory contains essential Python modules utilized in the notebooks. These modules facilitate the implementation of various functions and algorithms.

# data
The 'data' directory encompasses the input data utilized in the current research. It is organized into the following subdirectories:
- data_arpa: This directory contains the data recorded by ARPA stations.
- data_era5: Here, you will find the data downloaded from the ERA5 database.
- data_sensors: This directory holds the data measured by local sensors.


# use description
The following Python packages are required in order to run the code:
- numpy
- pandas
- sys
- pickle
- matplotlib
- seaborn
- datetime
- pygrib (maybe can generate some conflicts with tensorflow)
- random
- os
- copy
- scipy
- sklearn
- tensorflow

In order to setup the coding framework it is suggested to create a conda environment and work on jupyter lab (the present reasearch has been conducted with these tools).


Feel free to explore the project and leverage the provided resources for further analysis and development. If you have any questions or suggestions, please don't hesitate to reach out.
