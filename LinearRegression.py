import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf
import plotly.graph_objects as go

print("Downloading historical stock data...")

selected_ticker = input("Enter the stock ticker (e.g., AAPL, MSFT): ").upper()
#download 16 years of stock data of selected ticker from input
df = yf.download(selected_ticker, start="2010-01-01", end="2026-11-13", progress=False)

#if multi index columns occur in new yfinance version, handle it
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

print("Engineering features...")
#create simple techinical indicators(MA) as features
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_20"]= df["Close"].rolling(window=20).mean()

#drop rows with missing values caused by rolling windows
df = df.dropna()

#Create the target
#shift the close price 5 days into future to predict the next 5 days' closing price
df["Target"] = df["Close"].shift(-5)
df = df.dropna()  # Drop the last row which will have NaN target

#Define our inputs x and target y
x = df[["Close", "MA_5", "MA_20"]]
y = df["Target"]

print("Splitting into past and future sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = False)

print(f"Training on {len(x_train)} past days...")
print(f"Testing on {len(x_test)} future days...")

print("Traiing the linear regression model")
model = LinearRegression()
model.fit(x_train, y_train)

print("Making predictions on the future test data..")
predictions = model.predict(x_test)

#put actual vs predicted values into a clean result dataframe side by side
results_df = pd.DataFrame(
    {'Actual_Price': y_test.values, 'Predicted_Price': predictions}, index=y_test.index,
)

print('\n----Sample predictions (last 5 days)----')
print(results_df.tail())

#calculate mean absolute error 
mae = (results_df['Actual_Price'] - results_df['Predicted_Price']).abs().mean()
print(f'\nAverage pricing error MAE: ${mae:.2f} per share')

# Calculate the correct absolute percentage error for each row
results_df['Pct_Error'] = (
    (results_df['Actual_Price'] - results_df['Predicted_Price']).abs()
    / results_df['Actual_Price']
) * 100

# Calculate the average percentage error across the entire test set
mean_pct_error = results_df['Pct_Error'].mean()

print('\n--- Sample Prediction Percentage Error ---')
print(results_df.tail())

print(f'\nAverage Percentage Error: {mean_pct_error:.2f}% per share')

#visualise teh results with plotly
fig = go.Figure()

#actual price line
fig.add_trace(
    go.Scatter(
        x=results_df.index,
        y=results_df['Actual_Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='cyan', width = 2)
    )
)

#predicted price line
fig.add_trace(
    go.Scatter(
        x=results_df.index,
        y=results_df['Predicted_Price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange', width = 2, dash = 'dash')
    )
)

#update layout and show the plot
fig.update_layout(
    title=f'Actual vs Predicted Closing Prices for {selected_ticker}',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    template='plotly_dark',
    legend=dict(x=0.01, y=0.99)
)

#show the plot
fig.show()