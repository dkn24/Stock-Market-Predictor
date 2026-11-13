import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import plotly.graph_objects as go

print("Downloading historical stock data...")

selected_ticker = input("Enter the stock ticker (e.g., AAPL, MSFT, NVDA): ").upper()
df = yf.download(selected_ticker, start="2010-01-01", end="2026-01-01", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

print("Engineering features...")
# Expanded feature set to give Lasso more variables to evaluate & prune
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_20"] = df["Close"].rolling(window=20).mean()
df["MA_50"] = df["Close"].rolling(window=50).mean()
df["Volatility"] = df["Close"].rolling(window=20).std()
df["Price_Change"] = df["Close"].pct_change()

df = df.dropna()

# Target: predict 5 days into the future
df["Target"] = df["Close"].shift(-5)
df = df.dropna() 

# Define our broader feature inputs x and target y
x = df[["Close", "MA_5", "MA_20", "MA_50", "Volatility", "Price_Change"]]
y = df["Target"]

print("Splitting into past and future sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

print(f"Training on {len(x_train)} past days...")
print(f"Testing on {len(x_test)} future days...")

# --- FEATURE SCALING (Required for Lasso) ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Training the Lasso Regression model...")
# alpha controls regularization strength. Higher alpha = more sparsity (more coefficients driven to 0)
model = Lasso(alpha=0.01)
model.fit(x_train_scaled, y_train)

# Inspect sparsity: see which features Lasso kept vs dropped
print("\n--- Lasso Feature Weights (Sparsity Check) ---")
for feature, coef in zip(x.columns, model.coef_):
    if coef == 0:
        print(f"Dropped (Sparse): {feature}")
    else:
        print(f"Kept: {feature} with weight {coef:.4f}")

print("Making predictions on the future test data...")
predictions = model.predict(x_test_scaled)

results_df = pd.DataFrame(
    {'Actual_Price': y_test.values, 'Predicted_Price': predictions}, index=y_test.index,
)

print('\n----Sample predictions (last 5 days)----')
print(results_df.tail())

mae = (results_df['Actual_Price'] - results_df['Predicted_Price']).abs().mean()
print(f'\nAverage pricing error MAE: ${mae:.2f} per share')

results_df['Pct_Error'] = (
    (results_df['Actual_Price'] - results_df['Predicted_Price']).abs()
    / results_df['Actual_Price']
) * 100

mean_pct_error = results_df['Pct_Error'].mean()
print('\n--- Sample Prediction Percentage Error ---')
print(results_df.tail())

print(f'\nAverage percentage error: {mean_pct_error:.2f}% per share')

# Visualise results with Plotly
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=results_df.index,
        y=results_df['Actual_Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='cyan', width=2)
    )
)

fig.add_trace(
    go.Scatter(
        x=results_df.index,
        y=results_df['Predicted_Price'],
        mode='lines',
        name='Predicted Price (Lasso)',
        line=dict(color='orange', width=2, dash='dash')
    )
)

fig.update_layout(
    title=f'Lasso Regression: Actual vs Predicted Closing Prices for {selected_ticker}',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    template='plotly_dark',
    legend=dict(x=0.01, y=0.99)
)

fig.show()