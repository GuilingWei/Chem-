# 3. Support Vector Regressor
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
print("SVR R2 Score:", r2_score(y_test, y_pred_svr))
print("SVR MSE:", mean_squared_error(y_test, y_pred_svr))
