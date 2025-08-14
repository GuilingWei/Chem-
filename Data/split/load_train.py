import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load data
df = pd.read_csv("spectra_ld50.csv")

# 2. Extract features and target
inchikeys = df["InChIKey"]
X = df.drop(columns=["InChIKey", "LD50"]).values
y = df["LD50"].values

# Optional: Log-transform LD50 if distribution is skewed
# y = np.log1p(y)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

