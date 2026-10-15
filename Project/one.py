# AI fraud detection - creating a model from real world data
# You will be provided with the dataset of 8000 transaction records

import pandas as pd

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
