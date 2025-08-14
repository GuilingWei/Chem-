# 6. CNN

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNN input
X_train_cnn = X_train_scaled[..., np.newaxis]  # shape: (samples, 2520, 1)
X_test_cnn = X_test_scaled[..., np.newaxis]

# Build CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=8, activation='relu', input_shape=(2520, 1)),
    layers.MaxPooling1D(pool_size=4),
    layers.Conv1D(64, kernel_size=8, activation='relu'),
    layers.MaxPooling1D(pool_size=4),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train
model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)
