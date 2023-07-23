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

test_range = parameters['test_range']

train = df.iloc[:-test_range].copy()
test = df.iloc[-test_range:].copy()

test_months = [test.index[0], test.index[-1]]

train_diff = train.copy()
train_diff['new_cases_per_million'] = train['new_cases_per_million'] - train['new_cases_per_million'].shift(1)
train_diff.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

scaler.fit(train_diff)

train_scaled = scaler.transform(train_diff)

n_inputs = parameters['n_inputs']
n_features = 1

neurons_number = parameters['neurons_number']
l_rate = parameters['l_rate']
epochs_number = parameters['epochs_number']

generator = TimeseriesGenerator(train_scaled, train_scaled, length=n_inputs, batch_size=1)

train_diff['scaled'] = train_scaled

model = Sequential()

model.add(LSTM(neurons_number, activation='tanh', input_shape=(n_inputs, n_features)))
model.add(Dense(1))

opt = tf.keras.optimizers.Adam(learning_rate=l_rate)

model.compile(optimizer=opt, loss='mse')
model.summary()

model.fit(generator, epochs=epochs_number)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.title('Krzywa uczenia')
plt.ylabel('mse')
plt.xlabel('epoka')
plt.show()

save_name = parameters['save_name']

model.save('models/' + save_name)

train_predictions_diff = scaler.inverse_transform(model.predict(generator, batch_size=n_inputs))

train_diff['predictions'] = np.NaN
train_diff.iloc[n_inputs:, 2] = train_predictions_diff

train_predictions = []
train['predictions'] = np.nan

for i in range(len(train_predictions_diff)):
    train_predictions.append(train.iloc[i+n_inputs, 0] + train_predictions_diff[i][0])

train.iloc[n_inputs+1:, 1] = train_predictions

test_predictions = []

first_eval_batch = train_scaled[-n_inputs:]
current_batch = first_eval_batch.reshape((1, n_inputs, n_features))

for i in range(len(test)):
    
    current_pred = model.predict(current_batch)[0]
    
    test_predictions.append(current_pred) 
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)

test_true_predictions = []
test_true_predictions.append(train.iloc[-1,0] + true_predictions[0][0])

for i in range(test_range-1):
    test_true_predictions.append(test_true_predictions[-1]+true_predictions[i+1][0])
    
test['predictions'] = test_true_predictions

all_predictions = np.append(train_predictions, test_true_predictions)

df['Predictions'] = np.NaN
df.iloc[n_inputs+1:, 1] = all_predictions

train_rmse=sqrt(mean_squared_error(train.iloc[n_inputs+1:,0], train.iloc[n_inputs+1:,1]))
print(f'{train_rmse = }')

test_rmse=sqrt(mean_squared_error(test.iloc[:,0],test.iloc[:,1]))
print(f'{test_rmse = }')

r2 = r2_score(train.iloc[n_inputs+1:,0], train.iloc[n_inputs+1:,1])
print(f'{r2 = }')

dates_interval = parameters['dates_interval']

g = sns.lineplot(data=df, x = 'date', y = 'new_cases_per_million', label = 'dane')
sns.lineplot(data=df, x = 'date', y = 'Predictions', label = 'RNN')
plt.axvspan(*test_months, facecolor='grey', alpha=0.25)
plt.ylabel('dzienne zachorowania')
plt.xlabel('data')
plt.title('Dzienna liczba zachorowań na COVID-19 w Polsce na 1mln mieszkańców')
g.xaxis.set_major_locator(mdates.DayLocator(interval=dates_interval))
plt.legend()
plt.show()