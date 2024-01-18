#%%
#!/usr/bin/python3.10
#make sure python version is 3.10

import tensorflow as tf
import numpy as np




#Couldn't get pandas working idk why
#https://stackoverflow.com/questions/12332975/how-can-i-install-a-python-module-within-code
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("pandas")
install("matplotlib")
install("scikit-learn")

import pandas as pd
import matplotlib
import sklearn
from sklearn.linear_model import Ridge #Ridge Regression from scikit-learn



data = pd.read_csv('STL Science Center Weather Data - Weather_Clean.csv', index_col="DATE")
#print(data.tail())

#converts date column to datetime dtype
data.index = pd.to_datetime(data.index)
#print(data.dtypes)

#drop null/empty rows: data not null and data filled null
#print(data.isna().sum())
dataNN = data.dropna() 

dataFN = data.copy()
for col in dataFN.columns:
    dataFN[col] = dataFN[col].fillna(0)

#checks that the data fits within an expected range (ie negative precipitation) 
#print(dataNN.describe().transpose())
#print(dataNN.head(),data.tail())

#print(dataNN.loc["1995-06-23"])
#print(dataFN.describe().transpose())
#print(dataFN.index, dataNN.index)
#dataNN[["TMAX", "TMIN"]].plot() #doesn't work idk why

#this just checks how complete the data is (how many days out of each year in each year are recorded)
#print(dataNN.index.year.value_counts().sort_index())

#this adds a new column to the dataset that takes the next day's max temp 
dataNN["target"] = dataNN.shift(-1)["TMAX"]
#this removes the very last row in the dataset
dataNN = dataNN.iloc[:-1,:].copy()

#This is a Ridge Regression model
reg = Ridge(alpha=0.1)

predictors = list(dataFN.columns) #gets a list of the columns
train = dataNN.loc[:"2021-12-31"]
test = dataNN.loc["2022-01-01":] #splits into training/testing data
#I chose this date to ensure that there was at least a years worth of data to test

reg.fit(train[predictors], train["target"]) #this trains the model on the metrics, tries to predict the target
#the target is the next day's maximum temperature, so essentially we try to predict the high for the next day

#create nparray with predictions 
predictions = reg.predict(test[predictors])
#then combine into one dataframe
results = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
results.columns = ["actual", "predicted"]

print(results.head())