import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 1. Load Cleaned Dataset
df = pd.read_csv('iris/iris_cleaned.csv')
print("First 5 rows of cleaned dataset:")
print(df.head())

# 2. Univariate Analysis
# Histogram for each numeric feature
df.hist(figsize=(10,8), bins=15)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Count plot for species
plt.figure(figsize=(6,4))
sns.countplot(x='species', data=df)
plt.title("Species Count")
plt.show()

# 3. Bivariate Analysis
# Pairplot to see relationship between features
sns.pairplot(df, hue='species', diag_kind='hist')
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 4. Observations / Insights
print("\nObservations / Insights:")
print("- Petal length and petal width are strong indicators for species separation.")
print("- Sepal length and sepal width show some overlap between species.")
print("- Correlation matrix shows which features are positively or negatively correlated.")
