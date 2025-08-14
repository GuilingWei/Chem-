import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

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

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 4: Reshape the data for CNN input (1D convolution)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshape for Conv1D input
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Reshape for Conv1D input

# Step 5: Build the CNN Model
cnn_model = Sequential()
cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
cnn_model.add(MaxPooling1D(2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(1))  # For regression, output layer is a single unit

# Step 6: Compile the model
cnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the CNN model
cnn_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Step 8: Evaluate the CNN model
y_pred_cnn = cnn_model.predict(X_test_reshaped)

# Step 9: Evaluate performance using R2 and MSE
mse_cnn = mean_squared_error(y_test, y_pred_cnn)
r2_cnn = r2_score(y_test, y_pred_cnn)

print(f'Mean Squared Error (CNN): {mse_cnn}')
print(f'R-squared (CNN): {r2_cnn}')
