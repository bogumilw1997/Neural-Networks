import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from tensorflow import keras
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
from json import load

with open("data/parameters.json") as f:
    parameters = load(f)
    
df = pd.read_csv('data/owid-covid-data.csv', index_col = 'date', parse_dates= True)
df_poland = df.loc[df['location'] == 'Poland']
df_poland_nc = df_poland.loc[:, ['new_cases_per_million']]
df_poland_nc.index.freq = 'D'
df = df_poland_nc.loc[df_poland_nc.index >= '2021-09-01'].copy()

plt.rcParams["figure.figsize"] = [15, 8]
sns.set_theme(style="white", font_scale=1.5)

train = df.copy()
train_diff = train.copy()
train_diff['new_cases_per_million'] = train['new_cases_per_million'] - train['new_cases_per_million'].shift(1)
train_diff.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

scaler.fit(train_diff)

train_scaled = scaler.transform(train_diff)

n_inputs = parameters['n_inputs']
n_features = 1

prediction_range = parameters['prediction_range']

generator = TimeseriesGenerator(train_scaled, train_scaled, length=n_inputs, batch_size=1)

load_name = parameters['load_name']

model = keras.models.load_model('models/' + load_name)

train_predictions_diff = scaler.inverse_transform(model.predict(generator, batch_size=n_inputs))

train_predictions = []

for i in range(len(train_predictions_diff)):
    train_predictions.append(train.iloc[i+n_inputs, 0] + train_predictions_diff[i][0])

test_predictions = []

first_eval_batch = train_scaled[-n_inputs:]
current_batch = first_eval_batch.reshape((1, n_inputs, n_features))

for i in range(prediction_range):
    
    current_pred = model.predict(current_batch)[0]
    
    test_predictions.append(current_pred) 
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)

test_true_predictions = []
test_true_predictions.append(train.iloc[-1,0] + true_predictions[0][0])

for i in range(prediction_range-1):
    test_true_predictions.append(test_true_predictions[-1]+true_predictions[i+1][0])
    
all_predictions = np.append(train_predictions, test_true_predictions)
   
predictions_df = pd.DataFrame()
predictions_df['date'] = pd.date_range(start=df.index[n_inputs+1], periods=len(train)-(n_inputs+1)+prediction_range)
predictions_df['Predictions'] = all_predictions  
 
dates_interval = parameters['dates_interval']

g = sns.lineplot(data=df, x = 'date', y = 'new_cases_per_million', label = 'dane')
sns.lineplot(data=predictions_df, x = 'date', y = 'Predictions', label = 'RNN')
plt.ylabel('dzienne zachorowania')
plt.xlabel('data')
plt.title('Dzienna liczba zachorowań na COVID-19 w Polsce na 1mln mieszkańców')
g.xaxis.set_major_locator(mdates.DayLocator(interval=dates_interval))
plt.legend()
plt.show()