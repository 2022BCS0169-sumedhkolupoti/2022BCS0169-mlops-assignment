import pandas as pd
from sklearn.datasets import load_breast_cancer
import os
import sys

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Create data directory
os.makedirs('data', exist_ok=True)

def save_version(version_name, samples=None):
    if samples:
        df_v = df.iloc[:samples].copy()
    else:
        df_v = df.copy()
    
    path = 'data/breast_cancer.csv'
    df_v.to_csv(path, index=False)
    print(f"Dataset {version_name} created at {path} with {len(df_v)} samples")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'v2':
        save_version('v2') # Full dataset
    else:
        save_version('v1', 300) # Partial dataset (300 samples)
