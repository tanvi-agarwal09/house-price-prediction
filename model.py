import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# 1. Load dataset
df = pd.read_csv("data/Housing.csv")

# Clean & preprocess
df = df.drop_duplicates().dropna()

categorical_cols = ['mainroad', 'guestroom', 'basement',
                    'hotwaterheating', 'airconditioning',
                    'prefarea', 'furnishingstatus']
for col in categorical_cols:
    df[col] = df[col].str.strip().str.lower()

# Cap outliers
df = df[df['price'] <= df['price'].quantile(0.99)]
df = df[df['area'] <= df['area'].quantile(0.99)]

# Encode categorical
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,       # number of trees
    max_depth=10,           # limit depth to prevent overfitting
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
model.fit(X_train, y_train)

# Save model + feature names
with open("model/house_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": X.columns.tolist()}, f)

print("✅ Random Forest model trained and saved successfully.")
