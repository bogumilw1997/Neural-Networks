# Neural Networks

University project desined for predicting new COVID-19 cases using Recurrent Neural Network.

# Usage

First we need to download the `owid-covid-data.csv` file containing daily covid cases and put it into the `data` folder. You can download the file here: https://ourworldindata.org/explorers/coronavirus-data-explorer.

Folder contains 3 python scripts:
* `RNN_train.py` trains the RNN on the training set and evaluates predictions on a test set
* `RNN_load.py` loads trained model from `models` folder and makes the prediction with a set length
* `RNN_predict.py` takes all available data as a training set and makes a prediction with a set length

# Parameters

File `parameters.json` in `data` folder controls the model parameters:
* `test_range` controls  size of the test set in `RNN_train.py` script
* `n_inputs` is the number of past days used for predicting new cases at the next day 
* `neurons_number` is the amount of neaurons at the LSTM layer
* `l_rate` is the learing rate
* `epochs_number` is the number of epochs in a training process
* `prediction_range` is the number of predictions made by model (days)
* `dates_interval` is the density of data points at the resulting graph (days)
* `save_name` is the model name used to save the traind model
* `load_name` is the model name used to load the traind model
