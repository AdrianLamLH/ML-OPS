import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/processed', exist_ok=True)
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv('data/adult.data', header=None, names=column_names, sep=', ')
df = df.replace('?', np.nan)
df = df.dropna()
X = df.drop('income', axis=1)  # Replace 'target_column' with your actual target column
y = df['income']  # Replace 'target_column' with your actual target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed datasets
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Preprocessing completed successfully!")