#How much do transaction vary - two dataset can have same average but behave very differently
# for eg - customer grp A - 900 950 1000 1050 1100
# customer grp B - 100 500 1000 1500 1900 
#load the fraud detection dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("transaction8000.csv")

print(df.shape)

print(df.head())

amount = df['amount'].dropna()
#range = max - min - using grp A 1100 - 900 = 200 and using grp B 1900 - 100 = 1800 
GroupA = pd.Series([900,950,1000,1050,1100])
GroupB = pd.Series([100,500,1000,1500,1900])
print("GroupA range is",GroupA.max()-GroupA.min())
print("GroupB range is",GroupB.max()-GroupB.min())

Amount_range = amount.max()-amount.min()
print("Amount range is",Amount_range)

#Varience - measures house spreadout observation are around there mean
#calculation for the varience are done by -
#step 1 - calculate the mean
#step 2 - subtract mean from each observation
#step 3 - square each diff
#step 4 - calculate avg of square diff

print("GroupA population varience",GroupA.var(ddof = 0))
print("GroupB population varience",GroupB.var(ddof = 0))
print("amount population varience",amount.var(ddof = 0))

#Varience is useful but its values are squared so for that we use standard deviation
#Standard deviation is the square root of the varience
print("GroupA standard deviation",GroupA.std(ddof = 0))
print("GroupB standard deviation",GroupB.std(ddof = 0))
print("amount standard deviation",amount.std(ddof = 0))


