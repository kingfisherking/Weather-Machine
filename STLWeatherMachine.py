#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras

test_results = {}
column_names = ['Date','Precipitation', 'Snow', 'TempHigh', 'TempLow']
raw_data = pd.read_csv("../Weather Machine Learning/STL Science Center Weather Data - Weather_Clean.csv", names=column_names)
#raw_data.index = pd.to_datetime(raw_data.index)



#clean data
dataset = raw_data.dropna()

dataset["Date"] = pd.to_datetime(dataset["Date"], format='%Y-%m-%d')

#split data into training/testing
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#print(data.head())
print(dataset.head())


train_features = train_dataset.copy()
train_labels = train_features.pop('Date')

test_features = test_dataset.copy()
test_labels = test_features.pop('Date')

#train_labels.head()

normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(train_features))

#normalizer.mean.numpy()

weather_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(units=4)
])
weather_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

weatherMachine = weather_model.fit(np.asarray(train_features).astype(np.float32), np.asarray(train_labels).astype(np.float32), epochs=100, verbose=0)
results = weather_model.predict(test_features)

test_results['weather_model'] = weather_model.evaluate(np.asarray(test_features).astype(np.float32), np.asarray(test_labels).astype(np.float32), verbose=0)

#check = weather_model.predict()

# %%
