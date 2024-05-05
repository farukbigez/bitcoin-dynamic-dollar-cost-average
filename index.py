import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp

def dynamic_dca(starting_capital, base_investment, price_data):
    # Initialize DataFrame to store investment data
    investment_data = pd.DataFrame(index=price_data.index)
    investment_data['price'] = price_data['Adj Close']

    # Ensure all monetary and BTC amounts are treated as floating point
    investment_data['monthly_investment'] = float(base_investment)
    investment_data['btc_bought'] = investment_data['monthly_investment'] / investment_data['price']

    # Track remaining capital as floating point
    current_capital = float(starting_capital - base_investment)

    # Apply DCA strategy
    for i in range(1, len(investment_data)):
        # Calculate percentage change in price
        price_change = (investment_data['price'].iloc[i] / investment_data['price'].iloc[i - 1] - 1) * 100

        # Adjust investment based on price change
        if price_change < -10:  # If price drops by more than 10%
            investment = min(current_capital, investment_data['monthly_investment'].iloc[i - 1] * 1.50)
        elif price_change > 10:  # If price rises by more than 10%
            investment = max(0, investment_data['monthly_investment'].iloc[i - 1] * 0.50)
        else:
            investment = min(current_capital, base_investment)

        # Apply investment and calculate BTC bought
        investment_data.loc[investment_data.index[i], 'monthly_investment'] = float(investment)
        investment_data.loc[investment_data.index[i], 'btc_bought'] = investment / investment_data['price'].iloc[i]
        current_capital -= investment
        current_capital = max(current_capital, 0)  # Ensure no negative capital

    # Calculate total BTC and value
    investment_data['total_btc'] = investment_data['btc_bought'].cumsum()
    investment_data['total_value'] = investment_data['total_btc'] * investment_data['price']

    return investment_data

# Load data
btc_data = yf.download("BTC-USD", start="2018-01-01", end="2024-05-01", interval="1mo")
btc_data['Adj Close'].dropna(inplace=True)

# Run the simulation
results = dynamic_dca(10000, 1000, btc_data)

def calculate_returns(data):
    """ Calculate monthly and annual returns """
    data['monthly_return'] = data['total_value'].pct_change()
    data['annual_return'] = data['total_value'].pct_change(12)  # Assuming the data is monthly

def calculate_volatility(data):
    """ Calculate monthly and annual volatility """
    data['monthly_volatility'] = data['monthly_return'].rolling(window=12).std()
    data['annual_volatility'] = data['annual_return'].rolling(window=12).std()

def linear_regression_analysis(data):
    """ Perform linear regression on total value over time """
    model = LinearRegression()
    x = np.array(range(len(data))).reshape(-1, 1)  # Time as independent variable
    y = data['total_value'].values.reshape(-1, 1)  # Total value as dependent variable
    model.fit(x, y)
    return model.coef_[0][0], model.intercept_[0]  # Slope and intercept

def sharpe_ratio(data):
    """ Calculate Sharpe Ratio assuming risk-free rate is 0 for simplicity """
    return data['annual_return'].mean() / data['annual_return'].std()

def plot_correlation(data):
    """ Plot correlation between investment amounts and BTC price changes """
    price_changes = data['price'].pct_change()
    sns.scatterplot(x=price_changes, y=data['monthly_investment'])
    plt.xlabel('Price Change (%)')
    plt.ylabel('Investment Amount ($)')
    plt.title('Correlation between Price Changes and Investment Amounts')
    plt.show()
    print("Correlation Coefficient:", pearsonr(price_changes[1:], data['monthly_investment'][1:])[0])

def calculate_statistics(data):
    """Calculate various financial statistics."""
    data['cumulative_return'] = (1 + data['monthly_return']).cumprod() - 1
    annualized_return = np.power(1 + data['cumulative_return'].iloc[-1], 12 / len(data)) - 1
    cagr = ((data['total_value'].iloc[-1] / data['total_value'].iloc[0]) ** (12 / len(data))) - 1
    max_drawdown = (data['total_value'].cummax() - data['total_value']).max() / data['total_value'].cummax().max()
    data['drawdown'] = (data['total_value'].cummax() - data['total_value']) / data['total_value'].cummax()

    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"CAGR (Compound Annual Growth Rate): {cagr:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # T-tests
    t_stat, p_value = ttest_1samp(data['monthly_return'].dropna(), 0)
    print(f"Monthly Return T-Test: T-Stat={t_stat:.3f}, P-Value={p_value:.3f}")

def plot_drawdown(data):
    """Plot the drawdown over time."""
    plt.figure(figsize=(10, 6))
    plt.fill_between(data.index, data['drawdown'], color="red", step="pre", alpha=0.4)
    plt.title('Drawdown over Time')
    plt.ylabel('Drawdown')
    plt.xlabel('Date')
    plt.show()

# Calculate returns and volatilities
calculate_returns(results)
calculate_volatility(results)

# Calculate statistics
calculate_statistics(results)
plot_drawdown(results)

# Perform linear regression
slope, intercept = linear_regression_analysis(results)
print(f"Linear Regression Slope: {slope}, Intercept: {intercept}")

# Calculate Sharpe Ratio
sharpe = sharpe_ratio(results)
print(f"Sharpe Ratio: {sharpe}")

# Plot correlation
plot_correlation(results)


# Cross correlation, t
