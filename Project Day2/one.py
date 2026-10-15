#if i tell you 8000 transactions happened yesterday give me a one number representing the critical transactions
#we will be using method dropna() - it stores the transaction where no values are missing
#Then we calculate smallest and largest transaction - min and max and then sort it by [.sort_values().head(10) we are using 10 for last 10 values]

#do you think largest transaction will affect the values of our typical transaction  [mean() and median()]

# AI fraud detection - creating a model from real world data
# You will be provided with the dataset of 8000 transaction records

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('payment_fraud_dataset.csv')

print(df.shape)

print(df.head())

print(type(df))

print(df.dtypes)

#catogorical data - catogorical data represents catogories rather than measurements
#Amount,customer age,account age = numerical
#new device,fraud = binary/catagorical(0,1)
#transition id = identifier(t00001)

#Task - we have to find missing values in csv file [df.isnull().sum]
print(df.isnull().sum())

#Task -To find the duplicate data, use this: [df.duplicated().sum()]
print(df.duplicated().sum())

#Task - how many fraudulent transactions are there : df['fraud'].value_counts()
print(df['fraud'].value_counts())

#df['fraud'].value_counts(normalize=True)*100
print(df['fraud'].value_counts(normalize=True)*100)

#no of records , columns 
#create a table using column, type , etc
#find out missing values , duplicates, fraud numbers and fraud percentage

#[raw data > inspect > question > investigation > find problem > understand the data > analyse]

print("missing values per column:\n", df.isnull().sum())
print("duplicate values:\n", df.duplicated().sum())
print("fraud counts:\n", df['fraud'].value_counts())
print("fraud percentage:\n", df['fraud'].value_counts(normalize=True)*100)

mean_amount = df['amount'].mean()
print("mean amount:", mean_amount)

