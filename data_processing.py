import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("crime_data.csv")  # Replace with your file name

# Display first few rows
print(df.head())

# Drop missing values
df.dropna(inplace=True)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract useful features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Hour'] = df['Date'].dt.hour

# Visualize crime trends over years
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Year')
plt.title("Crime Count Per Year")
plt.xticks(rotation=45)
plt.show()