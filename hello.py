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