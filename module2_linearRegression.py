import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np 
#%matplotlib inline - for jupyter notebooks

#from pyodide.http import pyfetch

path='FuelConsumption.csv'

df = pd.read_csv(path)

# take a look at the dataset
print(df.head())

# summarize the data
print(df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

#We can plot each of these features:
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

#Now, let's plot each of these features against the Emission, to see how linear their relationship is:
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Plot CYLINDER vs the Emission, to see how linear is their relationship is:
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='RED')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

"""
Creating train and test dataset
Train/Test Split involves splitting the dataset into training and testing sets that are mutually exclusive. After which, you train
with the training set and test with the testing set. This will provide a more accurate evaluation on out-of-sample accuracy because
the testing dataset is not part of the dataset that have been used to train the model. Therefore, it gives us a better understanding
of how well our model generalizes on new data.

This means that we know the outcome of each data point in the testing dataset, making it great to test with! Since this data has not 
been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an 
out-of-sample testing.

Let's split our dataset into train and test sets. 80% of the entire dataset will be used for training and 20% for testing.
We create a mask to select random rows using np.random.rand() function:
"""

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

print(test)
print(msk)

#Simple Linear regression

# the dataset named train from above which contains 80% of randomly selected rows
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Modelling - using Scikit learn to model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']]) #This converts pandas column to an array
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

#plotting the outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

"""
Evaluation
We compare the actual values and predicted values to calculate the accuracy of a regression model.
Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:
    - Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
    - Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean Absolute Error because the focus
      is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
    - Root Mean Squared Error (RMSE).
    - R-squared is not an error, but rather a popular metric to measure the performance of your regression model.
      It represents how close the data points are to the fitted regression line. The higher the R-squared value, 
      the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
"""
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

"""
Lets see what the evaluation metrics are if we trained a regression model using the FUELCONSUMPTION_COMB feature.
Start by selecting FUELCONSUMPTION_COMB as the train_x data from the train dataframe, then select FUELCONSUMPTION_COMB as the test_x data from the test dataframe
"""

train_x_fuelConsumption=np.asanyarray(train[['FUELCONSUMPTION_COMB']])
test_x_fuelConsumption=np.asanyarray(test[['FUELCONSUMPTION_COMB']])
regr_x_fuelConsumption = linear_model.LinearRegression()


regr_x_fuelConsumption.fit(train_x_fuelConsumption, train_y)
print ('Coefficients: ', regr_x_fuelConsumption.coef_)
print ('Intercept: ',regr_x_fuelConsumption.intercept_)

#testing the data set and calculating predictions
test_y_fuelConsumption=regr_x_fuelConsumption.predict(test_x_fuelConsumption)

#evaluation
mean_absolute_error_fuelConsumption=np.mean(np.absolute(test_y_fuelConsumption-test_y))
residualSumOfSquares_fuelConsumption=np.mean((test_y_fuelConsumption-test_y)**2)
r2_square_fuelConsumption=r2_score(test_y_fuelConsumption,test_y)
print("Mean absolute error: %.2f" % mean_absolute_error_fuelConsumption)
print("Residual sum of squares (MSE): %.2f" % residualSumOfSquares_fuelConsumption)
print("R2-score: %.2f" % r2_square_fuelConsumption)