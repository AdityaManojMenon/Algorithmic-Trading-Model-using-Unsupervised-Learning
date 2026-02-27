from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import warnings
import statsmodels.api as sm
from sklearn.cluster import KMeans
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.ticker as mtick
import requests
from io import StringIO

warnings.filterwarnings("ignore")


wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(wiki_url, headers=headers, timeout=30)
response.raise_for_status()
sp500 = pd.read_html(StringIO(response.text))[0]
print(sp500)

symbols_list = sp500['Symbol'].unique().tolist()
symbols_list = [symbol.replace('.', '-') for symbol in symbols_list]

end_date = "2024-01-31"
start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 8)

df = yf.download(
    tickers=symbols_list,
    start=start_date,
    end=end_date,
    group_by="column",
    auto_adjust=False,
    progress=True,
)
# Ensure OHLCV is the first level and tickers are the second level
if isinstance(df.columns, pd.MultiIndex):
    if "High" not in df.columns.get_level_values(0) and "High" in df.columns.get_level_values(1):
        df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
else:
    # Single-ticker fallback: promote to MultiIndex columns
    df.columns = pd.MultiIndex.from_product([df.columns, [symbols_list[0]]])
# Drop tickers that have no price data at all
if isinstance(df.columns, pd.MultiIndex) and "Adj Close" in df.columns.get_level_values(0):
    valid_tickers = df["Adj Close"].dropna(axis=1, how="all").columns
    df = df.loc[:, (slice(None), valid_tickers)]

# Stack so index = (date, ticker)
df = df.stack(future_stack=True)           # future_stack=True avoids FutureWarning
df.index.names = ["date", "ticker"]
df.columns = df.columns.str.lower()
required_cols = {"high", "low", "open", "close", "adj close", "volume"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing OHLCV columns after download/stack: {sorted(missing_cols)}")

# ── Technical Indicators ─────────────────────────────────────────────────────
df["garman_klass_vol"] = (
    (np.log(df["high"]) - np.log(df["low"])) ** 2
) / 2 - (
    (2 * np.log(2) - 1)
    * (np.log(df["adj close"]) - np.log(df["open"])) ** 2
) / 2

df["rsi"] = df.groupby(level=1, group_keys=False)["adj close"].transform(
    lambda x: RSIIndicator(close=x, window=20).rsi()
)

def compute_bbands(stock_data):
    bb = BollingerBands(close=stock_data["adj close"], window=20)
    stock_data["bb_low"]  = bb.bollinger_lband()
    stock_data["bb_mid"]  = bb.bollinger_mavg()
    stock_data["bb_high"] = bb.bollinger_hband()
    return stock_data

df = df.groupby(level=1, group_keys=False).apply(compute_bbands)

def compute_atr(stock_data):
    atr = AverageTrueRange(
        high=stock_data["high"],
        low=stock_data["low"],
        close=stock_data["close"],
        window=14,
    ).average_true_range()
    return atr.sub(atr.mean()).div(atr.std())

df["atr"] = df.groupby(level=1, group_keys=False).apply(compute_atr)

def compute_macd(close):
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd()
    return macd.sub(macd.mean()).div(macd.std())

df["macd"] = df.groupby(level=1, group_keys=False)["adj close"].apply(compute_macd)

df["dollar_volume"] = (df["adj close"] * df["volume"]) / 1e6

# ── Aggregate to Monthly Level ───────────────────────────────────────────────
convert_columns = [
    c for c in df.columns.get_level_values(0).unique()
    if c not in ["dollar_volume", "high", "low", "close", "volume", "open"]
]
print(convert_columns)

dollar_volume_resampled = (
    df["dollar_volume"]
    .unstack("ticker")
    .resample("ME")          # Month End frequency
    .mean()
    .stack()
    .to_frame("dollar_volume")
)
dollar_volume_resampled.index.names = ["date", "ticker"]

other_columns_resampled = (
    df[convert_columns]
    .unstack("ticker")
    .resample("ME")
    .last()
    .stack()
)
other_columns_resampled.index.names = ["date", "ticker"]

data = pd.concat([dollar_volume_resampled, other_columns_resampled], axis=1)
data = data.dropna(how="all")

# 5-year rolling mean of dollar volume
data["dollar_volume"] = (
    data["dollar_volume"]
    .unstack("ticker")
    .rolling(window=5 * 12, min_periods=1)
    .mean()
    .stack()
)

data["dollar_volume_rank"] = data.groupby(level=0)["dollar_volume"].rank(ascending=False)
data = data[data["dollar_volume_rank"] < 150]
data = data.drop(["dollar_volume", "dollar_volume_rank"], axis=1)

# ── Monthly Returns ──────────────────────────────────────────────────────────
def calculate_returns(df):
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f"return_{lag}m"] = (
            df["adj close"]
            .pct_change(lag)
            .pipe(lambda x: x.clip(
                lower=x.quantile(outlier_cutoff),
                upper=x.quantile(1 - outlier_cutoff),
            ))
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
print(data)

# ── Fama-French 5-Factor Betas ───────────────────────────────────────────────
ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
ff = pd.read_csv(ff_url, compression="zip", skiprows=3)
ff = ff.rename(columns={"Unnamed: 0": "date"})
ff = ff[ff["date"].astype(str).str.fullmatch(r"\d{6}")]
ff["date"] = pd.to_datetime(ff["date"], format="%Y%m")
ff = ff.set_index("date")
ff = ff.apply(pd.to_numeric, errors="coerce")
ff = ff.dropna(how="all")
factor_data = ff.resample("ME").last().div(100)
factor_data.index.name = "date"

# BUG FIX: join return_1m with proper multi-index alignment
# Pivot return_1m so it has the same shape as factor_data needs
return_1m = data["return_1m"].unstack("ticker")          # (date, ticker) → wide

# Join factor columns onto each stock's monthly return
# Result: MultiIndex (date, ticker) with factor columns + return_1m
factor_data_expanded = (
    return_1m
    .stack()
    .to_frame("return_1m")
    .join(factor_data, how="left")
    .dropna()
)
factor_data_expanded.index.names = ["date", "ticker"]

# Keep only tickers with >= 10 observations
checker = factor_data_expanded.groupby(level="ticker").size()
valid_stocks = checker[checker >= 10].index
factor_data_expanded = factor_data_expanded[
    factor_data_expanded.index.get_level_values("ticker").isin(valid_stocks)
]
print(factor_data_expanded)

# ── Rolling OLS Factor Betas ─────────────────────────────────────────────────
def compute_rolling_ols(x):
    min_window = min(24, x.shape[0])
    rolling_model = RollingOLS(
        endog=x["return_1m"],
        exog=sm.add_constant(x.drop("return_1m", axis=1)),
        window=min_window,
        min_nobs=len(x.columns) + 1,
    )
    fitted_model = rolling_model.fit(params_only=True)
    params = fitted_model.params.drop("const", axis=1, errors="ignore")
    return params

betas = factor_data_expanded.groupby(level="ticker", group_keys=False).apply(
    compute_rolling_ols
)

# Shift by 1 month within each ticker to avoid look-ahead bias
month_adjust = betas.groupby(level="ticker").shift(1)

data = data.join(month_adjust, on=["date", "ticker"], how="left")

factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
# Fill missing betas with each ticker's mean
data.loc[:, factors] = data.groupby("ticker", group_keys=False)[factors].apply(
    lambda x: x.fillna(x.mean())
)

data = data.drop("adj close", axis=1)
data = data.dropna()
print(data)

# ── K-Means Clustering ───────────────────────────────────────────────────────
rsi_data = data[["rsi"]].dropna().copy()
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
rsi_data["cluster"] = kmeans.fit_predict(rsi_data[["rsi"]])

data = data.drop(columns=["cluster"], errors="ignore")
data = data.join(rsi_data["cluster"], how="left")
print(data)

# ── Visualise clusters ───────────────────────────────────────────────────────
start_plot_date = pd.to_datetime("2017-03-31")
end_plot_date   = pd.to_datetime("2024-01-31")
dates_to_plot   = pd.date_range(start=start_plot_date, end=end_plot_date, freq="ME")

colors = ["red", "green", "blue", "purple"]

# BUG FIX: use get_level_values().unique() instead of .levels[0] (avoids stale levels)
valid_dates = data.index.get_level_values(0).unique()

plt.style.use("ggplot")
for plot_date in dates_to_plot:
    if plot_date not in valid_dates:
        continue
    monthly_data = data.loc[plot_date]
    plt.figure(figsize=(12, 8))
    for cluster in range(4):
        clustered_data = monthly_data[monthly_data["cluster"] == cluster]
        plt.scatter(
            clustered_data["atr"],
            clustered_data["rsi"],
            color=colors[cluster],
            label=f"Cluster {cluster}",
            alpha=0.5,
        )
    plt.title(f"K-Means Clusters for {plot_date.strftime('%Y-%m-%d')}")
    plt.xlabel("ATR")
    plt.ylabel("RSI")
    plt.legend()
    plt.show()

# ── Select high-momentum cluster (highest median RSI) ───────────────────────
# Identify which cluster has the highest RSI centroid
cluster_rsi_means = data.groupby("cluster")["rsi"].mean()
top_cluster = int(cluster_rsi_means.idxmax())
print(f"Top momentum cluster: {top_cluster} (mean RSI = {cluster_rsi_means[top_cluster]:.2f})")

filtered_df = data[data["cluster"] == top_cluster].copy()
print(filtered_df)

# ── Build fixed_dates dict (stocks to hold each month) ──────────────────────
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(["date", "ticker"])

dates = filtered_df.index.get_level_values("date").unique().tolist()
fixed_dates = {
    date.strftime("%Y-%m-%d"): filtered_df.xs(date, level=0).index.tolist()
    for date in dates
}

# ── Portfolio Optimisation ───────────────────────────────────────────────────
def optimized_weights(prices, lower_bound=0):
    ret = expected_returns.mean_historical_return(prices, frequency=252)
    cov = risk_models.sample_cov(prices, frequency=252)
    ef = EfficientFrontier(
        expected_returns=ret,
        cov_matrix=cov,
        weight_bounds=(lower_bound, 0.1),
        solver="SCS",
    )
    ef.max_sharpe()
    return ef.clean_weights()

stocks = data.index.get_level_values("ticker").unique().tolist()

portfolio_start = (
    data.index.get_level_values("date").unique()[0] - pd.DateOffset(months=12)
).strftime("%Y-%m-%d")
portfolio_end = data.index.get_level_values("date").unique()[-1].strftime("%Y-%m-%d")

new_df = yf.download(
    tickers=stocks,
    start=portfolio_start,
    end=portfolio_end,
    group_by="column",
    auto_adjust=False,
)
print(new_df)

def extract_price_frame(price_df, field):
    if isinstance(price_df.columns, pd.MultiIndex):
        if field in price_df.columns.get_level_values(0):
            out = price_df[field]
            if isinstance(out.columns, pd.MultiIndex):
                out = out.droplevel(0, axis=1)
            return out
        if field in price_df.columns.get_level_values(1):
            return price_df.xs(field, axis=1, level=1)
    if field in price_df.columns:
        return price_df[field]
    return None

# Handle MultiIndex columns from yfinance and missing Adj Close
adj_close = extract_price_frame(new_df, "Adj Close")
if adj_close is None:
    adj_close = extract_price_frame(new_df, "Close")
if adj_close is None:
    adj_close = extract_price_frame(new_df, "Price")
if adj_close is None:
    raise KeyError("Adj Close/Close/Price not found in downloaded data")

return_df = np.log(adj_close).diff()

portfolio_df = pd.DataFrame()

for start_date_str in fixed_dates.keys():
    end_date_str = (
        pd.to_datetime(start_date_str) + pd.offsets.MonthEnd(0)
    ).strftime("%Y-%m-%d")
    cols = fixed_dates[start_date_str]

    # BUG FIX: use loop variables, not hardcoded dates
    optimization_start_date = (
        pd.to_datetime(start_date_str) - pd.DateOffset(months=12)
    ).strftime("%Y-%m-%d")
    optimization_end_date = (
        pd.to_datetime(start_date_str) - pd.DateOffset(days=1)
    ).strftime("%Y-%m-%d")

    # Filter to stocks available in adj_close
    cols_available = [c for c in cols if c in adj_close.columns]
    if len(cols_available) < 2:
        continue

    optimization_df = adj_close.loc[optimization_start_date:optimization_end_date, cols_available].dropna(
        axis=1, how="any"
    )
    if optimization_df.shape[1] < 2 or optimization_df.shape[0] < 30:
        continue

    lower_bound = 1 / (len(optimization_df.columns) * 2)
    try:
        weights = optimized_weights(prices=optimization_df, lower_bound=lower_bound)
    except Exception as e:
        print(f"Optimisation failed for {start_date_str}: {e}")
        continue

    weights_df = (
        pd.DataFrame(weights, index=pd.Series(0))
        .stack()
        .to_frame("weights")
        .reset_index(level=0, drop=True)
    )
    print(weights_df)

    temp_df = return_df.loc[start_date_str:end_date_str]
    temp_df = (
        temp_df.stack()
        .to_frame("returns")
        .reset_index(level=0)                       # keeps ticker in index
        .merge(weights_df, left_index=True, right_index=True)
        .reset_index()
        .rename(columns={"index": "Ticker", "level_0": "Date"})
        .set_index(["Date", "Ticker"])
    )
    temp_df["weighted_returns"] = temp_df["returns"] * temp_df["weights"]
    temp_df = (
        temp_df.groupby(level=0)["weighted_returns"]
        .sum()
        .to_frame("Strategy Returns")
    )
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

print(portfolio_df)

# ── Compare with S&P 500 ─────────────────────────────────────────────────────
spy = yf.download(
    tickers="SPY",
    start="2016-01-01",
    end=dt.date.today(),
    group_by="column",
    auto_adjust=False,
)

# Handle MultiIndex columns and missing Adj Close
spy_adj = extract_price_frame(spy, "Adj Close")
if spy_adj is None:
    spy_adj = extract_price_frame(spy, "Close")
if spy_adj is None:
    spy_adj = extract_price_frame(spy, "Price")
if isinstance(spy_adj, pd.DataFrame):
    spy_adj = spy_adj.iloc[:, 0]

spy_return = (
    np.log(spy_adj)
    .diff()
    .dropna()
    .to_frame("S&P500_returns")
)
spy_return.index.name = "Date"

portfolio_df = portfolio_df.merge(spy_return, left_index=True, right_index=True)

# Summary metrics + plot
# Cumulative returns
strategy_cumulative = np.exp(np.log1p(portfolio_df["Strategy Returns"]).cumsum()) - 1
spy_cumulative = np.exp(np.log1p(portfolio_df["S&P500_returns"]).cumsum()) - 1

# Final cumulative return
strategy_total = strategy_cumulative.iloc[-1]
spy_total = spy_cumulative.iloc[-1]

# Sharpe Ratio (annualized)
strategy_sharpe = (portfolio_df["Strategy Returns"].mean() / portfolio_df["Strategy Returns"].std()) * np.sqrt(252)
spy_sharpe = (portfolio_df["S&P500_returns"].mean() / portfolio_df["S&P500_returns"].std()) * np.sqrt(252)

print(f"Strategy Cumulative Return: {strategy_total:.1%}")
print(f"S&P 500 Cumulative Return:  {spy_total:.1%}")
print(f"Strategy Sharpe Ratio:      {strategy_sharpe:.2f}")
print(f"S&P 500 Sharpe Ratio:       {spy_sharpe:.2f}")

plt.style.use("ggplot")
portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1
portfolio_cumulative_return[:"2024-01-30"].plot(figsize=(16, 6))
plt.title("Unsupervised Learning Trading Strategy vs S&P 500")
plt.ylabel("Returns")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.tight_layout()
plt.show()