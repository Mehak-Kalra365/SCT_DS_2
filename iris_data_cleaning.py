import os
import pandas as pd

# Folder jahan cleaned CSV save karna hai
folder_path = 'iris'

# Agar folder exist nahi karta, to create kar do
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 1. Load Dataset
df = pd.read_csv('iris/bezdekIris.data')
print("First 5 rows:")
print(df.head())

# 2. Data Cleaning
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Rename columns
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Convert species to category
df['species'] = df['species'].astype('category')

# 3. Save cleaned dataset
cleaned_file_path = os.path.join(folder_path, 'iris_cleaned.csv')
df.to_csv(cleaned_file_path, index=False)
print("\nCleaned dataset saved as", cleaned_file_path)
