import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("data.csv")

# Explore
print(df.head())
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Correct filling
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Groupby
print(df.groupby("Sex")["Survived"].mean())
print(df.groupby("Pclass")["Survived"].mean())

# NumPy
age_array = np.array(df["Age"])
print("Mean Age:", np.mean(age_array))
print("Std Age:", np.std(age_array))

# Visualizations
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.show()

# Correlation heatmap (FIXED)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
