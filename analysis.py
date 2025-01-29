# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv('credit_card_fraud_detection_data.csv')

# Display the first few rows of the dataset
print(df.head())

# ---------------- STEP 1: Data Exploration ---------------- #

# Get dataset shape (number of rows and columns)
print("\nDataset Shape:")
print(df.shape)

# Display dataset columns
print("\nDataset Columns:")
print(df.columns)

# Display data types of each column
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Display summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# ---------------- STEP 2: Data Cleaning ---------------- #

# Convert 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
print("\nTimestamp Column Type After Conversion:")
print(df['Timestamp'].dtype)  # Should be datetime64[ns]

# Convert 'Transaction Amount' to numeric (handling special characters like $ or ,)
df['Transaction Amount'] = df['Transaction Amount'].replace('[\$,]', '', regex=True).astype(float)

# Handle missing values:
# - Fill 'Touch Dynamics' and 'Gesture Dynamics' with their median
df['Touch Dynamics'].fillna(df['Touch Dynamics'].median(), inplace=True)
df['Gesture Dynamics'].fillna(df['Gesture Dynamics'].median(), inplace=True)

# ---------------- STEP 3: Exploratory Data Analysis (EDA) ---------------- #

# Convert 'Is Fraud' column to numeric (Yes → 1, No → 0)
df['Is Fraud'] = df['Is Fraud'].map({'Yes': 1, 'No': 0})

# 1️⃣ Count of fraudulent vs non-fraudulent transactions
fraud_counts = df['Is Fraud'].value_counts()
print("\nFraudulent vs Non-Fraudulent Transaction Counts:")
print(fraud_counts)

# Plot fraud distribution with percentage labels
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Is Fraud', data=df, palette=['blue', 'red'])
plt.title('Distribution of Fraudulent Transactions')

# Add percentage labels
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.2f}%'
    ax.annotate(percentage, (p.get_x() + 0.3, p.get_height() + 50), ha='center')

plt.show()

# 2️⃣ Summary statistics for 'Transaction Amount' by fraud status
print("\nTransaction Amount Statistics by Fraud Status:")
print(df.groupby('Is Fraud')['Transaction Amount'].describe())

# 3️⃣ Extract hour from the timestamp for time-based analysis
df['Hour'] = df['Timestamp'].dt.hour

# Plot transaction count by hour (Fraud vs Non-Fraud)
plt.figure(figsize=(12, 6))
sns.countplot(x='Hour', hue='Is Fraud', data=df, palette=['blue', 'red'])
plt.title('Number of Transactions by Hour')
plt.xlabel("Hour of the Day")
plt.ylabel("Transaction Count")
plt.show()

# 4️⃣ Count of transactions by merchant category (Top 10 for clarity)
plt.figure(figsize=(12, 6))
top_merchant_categories = df['Merchant Category'].value_counts().nlargest(10).index
sns.countplot(y='Merchant Category', hue='Is Fraud', 
              data=df[df['Merchant Category'].isin(top_merchant_categories)],
              palette=['blue', 'red'])
plt.title('Top 10 Merchant Categories by Fraud Status')
plt.show()

# 5️⃣ Count of unique devices and IPs
print("\nNumber of Unique Devices:", df['Device ID'].nunique())
print("Number of Unique IP Addresses:", df['IP Address'].nunique())

# 6️⃣ Top 5 IPs by transaction count
ip_counts = df['IP Address'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=ip_counts[:5].index, y=ip_counts[:5].values, palette='coolwarm')
plt.title('Top 5 IPs by Transaction Count')
plt.xlabel("IP Address")
plt.ylabel("Transaction Count")
plt.show()

# ---------------- STEP 4: Correlation Analysis ---------------- #

# Select only numeric columns automatically
num_cols = df.select_dtypes(include=['number']).columns

# Compute correlation matrix
corr_matrix = df[num_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
