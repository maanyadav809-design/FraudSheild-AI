# Statistics for AI — Fraud Detection: Student Project

You are a junior data scientist at a digital payments company. Your task is to investigate transaction data statistically and build an AI system that flags suspicious transactions.

## Tasks

### 1. Data Investigation
Find dataset size, data types, missing values, duplicate records, fraud counts/percentages, and variable types.

### 2. Typical Transaction
Calculate mean, median and mode of transaction amount. Plot its distribution. Decide which measure best represents a typical transaction.

### 3. Variability
Calculate minimum, maximum, range, variance and standard deviation. Explain what the standard deviation means.

### 4. Outliers
Calculate Q1, median, Q3 and IQR. Identify outliers and create a box plot. Explain why an outlier is not automatically fraud.

### 5. Z-Score
Calculate transaction z-scores. Find the 10 most unusual transactions. Explain what the z-score means.

### 6. Correlation
Investigate relationships between amount, failed attempts, new device, location change, account age and fraud. Create a correlation matrix. Explain why correlation does not prove causation.

### 7. Probability
Calculate P(Fraud), P(Fraud | New Device), and P(Fraud | Failed Attempts >= 2).

### 8. Bayes Theorem
Calculate P(Fraud | New Device) using Bayes theorem. Compare it with the direct calculation. Explain the difference between P(A|B) and P(B|A).

### 9. Model Evaluation
Create a naive model that predicts every transaction as legitimate. Calculate accuracy, precision, recall and F1. Explain why high accuracy can still be useless.

### 10. AI Model
Build a Logistic Regression classifier. Handle missing values, split the dataset, train the model, predict classes and probabilities, and evaluate it with a confusion matrix and classification report.

## Final Challenge
Analyze:
- Amount: ₹52,000
- Time: 03:12 AM
- New Device: Yes
- Location Changed: Yes
- Failed Attempts: 5
- Previous Transactions: 0
- Account Age: 45 days

Prepare a one-page Fraud Investigation Report explaining whether the transaction should be flagged.
