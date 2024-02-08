import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#loading data
file = 'Weather Machine Learning/STL Science Center Weather Data - Weather_Clean.csv'
column_names = ['Date','Precipitation', 'Snow', 'TemperatureHigh', 'TemperatureLow']
data = pd.read_csv(file, names=column_names)

#clean data
data = data.dropna(axis=0)
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data['Date'] = data['Date'].dt.dayofyear

#splitting data into training and testing
features = data[['Date']]                            # what I want as input
targets = data[['Precipitation', 'Snow', 'TemperatureHigh', 'TemperatureLow']]       #what I want to predict
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=1, shuffle=True)

#normallizing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#build model
weather_machine = LinearRegression()
weather_machine.fit(X_train_scaled, y_train)

#playing around with different dates, I'll probably print out some sort of calendar prediction
test_date = pd.to_datetime('2024-06-23').dayofyear
test_data = scaler.transform([[test_date]])
predictions = weather_machine.predict(test_data)

print(
    f'Predicted Precipitation: {predictions[0][0]}',
    f'Predicted Snow: {predictions[0][1]}',
    f'Predicted Temperature (High): {predictions[0][2]}',
    f'Predicted Temperature (Low): {predictions[0][3]}'
)

