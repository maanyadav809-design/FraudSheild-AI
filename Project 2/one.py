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

#we have to find fraud vs legit transactions
df.groupby('fraud')[('amount')].agg(['count', 'mean', 'median', 'std'])

#possible obsv will e sample size is small for fraud transactions, mean is higher than median for fraud transactions, 
# high variability in fraud transactions,
# std is higher for fraud transactions.

#we will be using seaborn for visualising 
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='fraud', y='amount')
plt.xlabel("fraud 0=legit, 1=fraud")
plt.ylabel("transaction amount")
plt.title("fraud status vs legit transactions")
plt.show()

#now we have to calc weighted mean
#for eg the company has 10 customers whose avg transaction is 500 & 90 customers whose avg trans is 2k
#we use weighted mean bcus the grps contain difff no of customers so the formula will be 10 * 500+90 * 2000 / 100

group_mean=np.array([500, 2000])
group_count=np.array([10, 90])
weighted_mean = np.average(group_mean, weights=group_count)
print(f"Weighted mean transaction: {weighted_mean}")



