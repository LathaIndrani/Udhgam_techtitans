import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

file_path = "C:/Users/kavya/Downloads/cleaned_data.csv"  # Update path if needed
df = pd.read_csv(file_path)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df['DDate'] = pd.to_datetime(df['DDate'], errors='coerce')
df = df.dropna(subset=['DDate'])  # Remove rows with missing dates

df['Year'] = df['DDate'].dt.year
df['Month'] = df['DDate'].dt.month
df['Day'] = df['DDate'].dt.day

df['AvgPrice'] = (df['MinPrice'] + df['MaxPrice']) / 2

features = ['AmcCode', 'YardCode', 'CommCode', 'VarityCode', 'Arrivals', 'Year', 'Month', 'Day']
X = df[features]

df['MinPrice'] = np.log1p(df['MinPrice'])
df['MaxPrice'] = np.log1p(df['MaxPrice'])

y_min = df['MinPrice']
y_max = df['MaxPrice']

X_train = X[X['Year'] < 2025]
y_min_train = y_min[X['Year'] < 2025]
y_max_train = y_max[X['Year'] < 2025]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model_min = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
model_max = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model_min.fit(X_train_scaled, y_min_train)
model_max.fit(X_train_scaled, y_max_train)

y_min_pred = model_min.predict(X_train_scaled)
y_max_pred = model_max.predict(X_train_scaled)

mse_min = mean_squared_error(y_min_train, y_min_pred)
mse_max = mean_squared_error(y_max_train, y_max_pred)

print(f"MSE for MinPrice: {mse_min}")
print(f"MSE for MaxPrice: {mse_max}")

X_2025 = X.copy()
X_2025['Year'] = 2025  # Set year to 2025
X_2025_scaled = scaler.transform(X_2025)

predicted_min_price = model_min.predict(X_2025_scaled)
predicted_max_price = model_max.predict(X_2025_scaled)

predicted_min_price = np.expm1(predicted_min_price)
predicted_max_price = np.expm1(predicted_max_price)

Q1 = np.percentile(predicted_max_price, 25)
Q3 = np.percentile(predicted_max_price, 75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = max(0, Q1 - 1.5 * IQR)

predicted_max_price = np.clip(predicted_max_price, lower_bound, upper_bound)

predicted_avg_price = (predicted_min_price + predicted_max_price) / 2

predictions_df = df.copy()
predictions_df['Year'] = 2025
predictions_df['MinPrice'] = predicted_min_price
predictions_df['MaxPrice'] = predicted_max_price
predictions_df['AvgPrice'] = predicted_avg_price

predictions_df = predictions_df[['Year', 'Month', 'Day', 'AmcCode', 'AmcName', 'YardCode', 'YardName',
                                 'CommCode', 'CommName', 'VarityCode', 'VarityName', 'Arrivals',
                                 'MinPrice', 'MaxPrice', 'AvgPrice']]

# Save the predicted data to a CSV file
output_file = "/mnt/data/predicted_crop_prices_evaluation.csv"
predictions_df.to_csv(output_file, index=False)
print(f"Predictions saved to: {output_file}")
