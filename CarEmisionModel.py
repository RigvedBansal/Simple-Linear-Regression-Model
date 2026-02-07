#Loading data
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Indian_Car_Emissions_1500.csv")
cdf = df[['ENGINE_SIZE','CYLINDERS','FUEL_CONSUMPTION_COMB','CO2_EMISSIONS']]
viz = cdf[['CYLINDERS','ENGINE_SIZE','FUEL_CONSUMPTION_COMB','CO2_EMISSIONS']]
viz.hist()
plt.show()

#Graphical and visual representation (Scatter Plots)
plt.scatter(cdf.FUEL_CONSUMPTION_COMB, cdf.CO2_EMISSIONS,  color='blue')
plt.xlabel("FUEL_CONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
plt.scatter(cdf.ENGINE_SIZE, cdf.CO2_EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)
plt.show()
plt.scatter(cdf.CYLINDERS,cdf.CO2_EMISSIONS, color ='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt. xlim(0,27)
plt.show()


#Exploring the relation between Engine Size and CO2 Emissions
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np

print("Data for Engine Size vs Emision regression")
X = cdf.ENGINE_SIZE.to_numpy()
y = cdf.CO2_EMISSIONS.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
type(X_train), np.shape(X_train), np.shape(X_train)
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
y_pred = regressor.predict(X_test.reshape(-1,1))

#Finding Errors and R2 for current model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred))

#Exploring the relation between Fuel Consumption and CO2 Emissions
print("Data for Fuel Consumption vs Emission regression")
X = cdf.FUEL_CONSUMPTION_COMB.to_numpy()
y = cdf.CO2_EMISSIONS.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
type(X_train), np.shape(X_train), np.shape(X_train)
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Fuel Consumption")
plt.ylabel("Emission")
plt.show()
y_pred = regressor.predict(X_test.reshape(-1,1))

#Finding Errors and R2 for current model
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred))