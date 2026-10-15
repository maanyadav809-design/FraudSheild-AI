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

df = pd.read_csv('transactions8000.csv')

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

amount = df['amount'].dropna()
mean_amount = round(amount.mean(), 2)
print("meantrasactionamount is", mean_amount)
#after this change a amount to very large number and find out thus it affect mean.It shows that mean is sensitive to extreme values

amount_meadian = amount.median()
#we will be using matplotlib libary as plt 
plt.figure(figsize=(10,5))
sns.histplot(amount,bins=50)
plt.xlabel("transaction amount")
plt.ylabel("no of transaction")
plt.title("distribution of transaction amount")
plt.show()
#our transaction data is not evenly distributed a relitvely small number of large transaction are pulling the mean upward
# median - what if instead of calculating an avg we simply sort all the transactions and pick the middle one
median_amount = amount.median(),2
print("the median amount is",median_amount)
#mean vs median - just compare
#mode - mode means the most common value - what transaction amount occurs most frequently
mode_amount = amount.mode()
print("the mode amount is",mode_amount)
#create transaction catogories 
bins = [0,500,1000,2000,5000,10000,50000,float("inf")]
labels = ["0-500","500-1000",]
df["amount range" = pd.cut(df["amount"]),bins=bins,labels=labels]

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



