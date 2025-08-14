import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the combined data (IR spectra + EFPC + LD50)
df = pd.read_csv('combined_features.csv')

# Extract features (excluding InChIKey and LD50)
features = [col for col in df.columns if col != 'InChIKey' and col != 'LD50']
X = df[features].values
y = df['LD50'].values

# Step 1: Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

