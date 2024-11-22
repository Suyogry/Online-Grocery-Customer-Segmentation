
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the raw dataset."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Clean and preprocess the data."""
    # Drop missing values
    data = data.dropna()

    # Standardize numeric features
    scaler = StandardScaler()
    numeric_features = ['PurchaseAmount', 'OrderFrequency', 'Age']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data

if __name__ == "__main__":
    # Load raw data
    data = load_data("data/raw_data.csv")

    # Preprocess data
    processed_data = preprocess_data(data)

    # Save processed data
    processed_data.to_csv("data/processed_data.csv", index=False)
