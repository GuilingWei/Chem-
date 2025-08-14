# 7. Plot Predictions

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.4, label="RF")
plt.scatter(y_test, y_pred_xgb, alpha=0.4, label="XGBoost")
plt.scatter(y_test, y_pred_cnn, alpha=0.4, label="CNN")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel("True LD50")
plt.ylabel("Predicted LD50")
plt.title("Model Predictions Comparison")
plt.legend()
plt.tight_layout()
plt.show()
