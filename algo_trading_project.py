from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader.data as web
import yfinance as yf
import pandas_ta as ta
import warnings
import statsmodels.api as sm
import sklearn as skp
from sklearn.cluster import KMeans
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.ticker as mtick
warnings.filterwarnings("ignore")


#Downloading the s&p500 data from wikipedia
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
#showing what the s&p 500 looks like this step is not necessary
print(sp500)

#Creating a unique list of all the symbols in the S&P 500
symbols_list = sp500['Symbol'].unique().tolist() 
#specifying the end data of the dataset
end_date = "2024-01-31"
#The start of the year is going to be 8 year before the end date
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

#This will download the S&P500 consistuents between the start and end dates.
#replacing dot with hyphen
symbols_list = [symbol.replace('.', '-') for symbol in symbols_list]  
df = yf.download(tickers = symbols_list,start = start_date,end = end_date)

#change the matrix so it is more suitable and data can be accessed more easily and efficiently 
df = df.stack()
df.index.names = ["date","ticker"]
df.columns = df.columns.str.lower()


############################################### Calculate Features and technical indicators of stock #######################################################


# Garman-Klass Volatility Formula
df["garman_klass_vol"] = ((np.log(df["high"]) - np.log(df["low"]))**2) / 2 - ((2 * np.log(2) - 1) * (np.log(df["adj close"]) - np.log(df["open"]))**2) / 2

# RSI Calculation
df["rsi"] = df.groupby(level=1, group_keys=False)["adj close"].transform(lambda x: RSIIndicator(close=x, window=20).rsi())


# Bollinger Bands Calculation
def compute_bbands(stock_data):
    bb = BollingerBands(close=stock_data["adj close"], window=20)
    stock_data["bb_low"] = bb.bollinger_lband()
    stock_data["bb_mid"] = bb.bollinger_mavg()
    stock_data["bb_high"] = bb.bollinger_hband()
    return stock_data

df = df.groupby(level=1, group_keys=False).apply(compute_bbands)


# ATR function requires three columns as input
def compute_atr(stock_data):
    atr = AverageTrueRange(high=stock_data["high"], low=stock_data["low"], close=stock_data["close"], window=14).average_true_range()
    if atr.isnull().any():
        print("ATR calculation resulted in NaNs")
    return atr.sub(atr.mean()).div(atr.std())

df["atr"] = df.groupby(level=1, group_keys=False).apply(compute_atr)


# Calculate MACD
def compute_macd(close):
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd()
    return macd.sub(macd.mean()).div(macd.std())

df["macd"] = df.groupby(level=1, group_keys=False)["adj close"].apply(compute_macd)


# Calculate dollar_volume
df["dollar_volume"] = (df["adj close"] * df["volume"]) / 1e6


##################################### Aggregate to Monthly Level and filter the top 150 most liquid stocks for each month ########################################


#Identify columns to convert and store it as a list
convert_columns = [c for c in df.columns.get_level_values(0).unique() if c not in ["dollar_volume", "high", "low", "close", "volume", "open"]]
print(convert_columns)

#unstacking will give the dollar_volume of a given company over a specificed period of time if not specificed for each corresponding date column value.
#here we are finding the monthly average then restack the data and add it to the dataframe as a column with the monthly averages. And we take the last adjusted value for technical indicators.
# Unstack and resample dollar_volume separately
dollar_volume_resampled = df["dollar_volume"].unstack("ticker").resample('M').mean().stack("ticker").to_frame("dollar_volume")
# Unstack and resample other columns
other_columns_resampled = df[convert_columns].unstack().resample("M").last().stack()
#Combine the results
data = pd.concat([dollar_volume_resampled, other_columns_resampled], axis=1).dropna()

#Calculating rolling mean for all dollar volume over the 5 year period so data.loc is very important
data["dollar_volume"] = (data.loc[:,"dollar_volume"].unstack("ticker").rolling(window=5*12, min_periods=1).mean().stack())
#Rank by dollar volume
data["dollar_volume_rank"] = data.groupby(level=0)["dollar_volume"].rank(ascending=False)
#filter based on top 150 dollar_volume
filtered_data = data[data["dollar_volume_rank"] < 150]
data = filtered_data
#removing the dollar_volume and dollar_volume_ranked columns since it is not needed after filtering the top 150 comapnies
data = data.drop(["dollar_volume", "dollar_volume_rank"], axis=1)

##### Calculate Monthly returns in different time horizons as features #####

# Define the calculate_returns function
def calculate_returns(df):
    outlier_cutoff = 0.005
    # Different month levels for return to track momentum over time of each stock
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f"return_{lag}m"] = (
            df["adj close"]
            .pct_change(lag)
            .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1 - outlier_cutoff)))
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
print(data)


########################################## Download Fama-French Factors and Calculate Rolling Factor Betas. #######################################################



# This datareader function reads data from various online sources and the precise dataset name is provided to fetch the information from.
# Index 0 is used to values of the first key in the dictionary which are the values of the factors by month. Index 1 is by year which we don't need.
factor_data = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start = "2010")[0]
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample("M").last().div(100)
factor_data.index.name = "date"

# Join the factor data with the calculated returns
try:
    factor_data = factor_data.join(data["return_1m"]).sort_index()
    print(factor_data)
except KeyError as e:
    print(f"KeyError: {e} - 'return_1m' column is not found in the DataFrame.")


checker = factor_data.groupby(level = 1).size()
valid_stocks = checker[checker >= 10]
factor_data = factor_data[factor_data.index.get_level_values("ticker").isin(valid_stocks.index)]
#Now that I am done removing the stocks which have data I can create the 2 year rolling factors betas by defining a function 
#endog is the dependent variable for the regression (y axis) and the exog is the independent variables that I specify as all the other variables except the dependent variable(return_1m).
#To achieve this we create a constant for all the columns except return_1m which we can drop later so we don't add redundant data to the main table. 
#window set the size of the rolling window for the regression and we need to make sure it take into account window can take data point less than 24
#min_nobs is the minimum number of observations required to perform the regression to ensure there are enough data points to estimate all the coefficients and constant terms.
#params_only indicates only the parameter cofficient in regression model are needed so this makes computation faster.
#Extracts the regression parameters from the fitted model 

def compute_rolling_ols(x):
    min_window = min(24, x.shape[0])
    rolling_model = RollingOLS(endog=x["return_1m"], exog=sm.add_constant(x.drop("return_1m", axis=1)), window=min_window, min_nobs=len(x.columns) + 1)
    fitted_model = rolling_model.fit(params_only=True)
    params = fitted_model.params.drop("const", axis=1, errors='ignore')
    return params

betas = factor_data.groupby(level="ticker", group_keys=False).apply(compute_rolling_ols)

month_adjust = betas.groupby("ticker").shift()

data = data.join(month_adjust, on=['date', 'ticker'], how='left')

factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
data.loc[:,factors] = data.groupby("ticker",group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
#drop all na values in the dataframe
data = data.drop("adj close",axis= 1)
data = data.dropna()
#printing the dataframe combined with the coresponding to rolling average betas
print(data)
# Done with data manipulation and perparing the data now it is time to analyze the data using Kmean clustering  (unsupervised learning) 


################################# For Each month fit K-Mean Clustering algorithm to group similar assets based on their features #################################


#We didn't apply normalization to rsi so we can see the cluster trend with respect to the rsi (dependent variable).
rsi_data = data[['rsi']].dropna()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
rsi_data['cluster'] = kmeans.fit_predict(rsi_data[['rsi']])

# Assign the clusters back to the main dataframe
data = data.drop(columns=['cluster'], errors='ignore')  # Remove existing 'cluster' column if exists
data = data.join(rsi_data['cluster'], how='left')

# Show the dataframe with KMean clusters unsupervised learning
print(data)


# Visualize the clusters for each month and can get the cluster the model allocates to have the upper bounds of rsi which we are using as the momentum indicator in our project.
start_plot_date = pd.to_datetime("2017-03-31")
end_plot_date = pd.to_datetime("2024-01-31")

dates_to_plot = pd.date_range(start=start_plot_date, end=end_plot_date, freq='M')

# Define a colormap with solid colors for the clusters
colors = ['red', 'green', 'blue', 'purple']
cmap = plt.cm.get_cmap('viridis', 4)
cmap = cmap(np.linspace(0, 1, 4))
plt.style.use("ggplot")

for plot_date in dates_to_plot:
    if plot_date not in data.index.levels[0]:
        continue

    monthly_data = data.loc[plot_date]
    plt.figure(figsize=(12, 8))
    for cluster in range(4):
        clustered_data = monthly_data[monthly_data['cluster'] == cluster]
        plt.scatter(clustered_data['atr'], clustered_data['rsi'], color=colors[cluster], label=f'Cluster {cluster}', alpha=0.5)
    plt.title(f'K-Means Clusters for {plot_date.strftime("%Y-%m-%d")}')
    plt.xlabel('ATR')
    plt.ylabel('RSI')
    #plt.legend()
    #plt.show()

filtered_df = data[data["cluster"] == 3].copy()
print(filtered_df)

#I am going to create a dictionary which will tell use what stocks to invest in the next month based on rsi being a strong indicator of momentum in our model.

#Reset and adjust index
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(["date", "ticker"])

#Create a dictionary mapping each unique date to its corresponding tickers
dates = filtered_df.index.get_level_values("date").unique().tolist()
fixed_dates = {date.strftime("%Y-%m-%d"): filtered_df.xs(date, level=0).index.tolist() for date in dates}


###################################################### Portfolio optimization function ########################################################################


#252 days is one year of trading data
def optimized_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices, frequency=252)
    cov = risk_models.sample_cov(prices, frequency=252)
    ef = EfficientFrontier(expected_returns=returns, cov_matrix=cov, weight_bounds=(lower_bound, 0.1), solver="SCS")
    weights = ef.max_sharpe()
    return ef.clean_weights()

stocks = data.index.get_level_values("ticker").unique().tolist()
# Download stock data for the required date range
new_df = yf.download(tickers=stocks, start=(data.index.get_level_values("date").unique()[0] - pd.DateOffset(months=12)).strftime('%Y-%m-%d'),end=data.index.get_level_values("date").unique()[-1].strftime('%Y-%m-%d'))
print(new_df)
return_df = np.log(new_df["Adj Close"]).diff()

portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    cols = fixed_dates[start_date]
    optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    #we are calculating the weights of each stock in the monthly optimization dataframe.
    optimization_df = new_df["2016-04-01":"2017-03-31"]["Adj Close"][fixed_dates["2017-04-01"]]
    lower_bound = 1/(len(optimization_df.columns)*2)

    weights = optimized_weights(prices = optimization_df,lower_bound=lower_bound)
    weights = pd.DataFrame(weights,index = pd.Series(0))
    temp_df = return_df[start_date:end_date]
    temp_df = temp_df.stack().to_frame("returns").reset_index(level=0).merge(weights.stack().to_frame("weights").reset_index(level=0,drop=True),left_index = True,right_index = True).reset_index().set_index(["Date","Ticker"])
    temp_df["weighted_returns"] = temp_df["returns"]*temp_df["weights"] 
    temp_df = temp_df.groupby(level = 0)["weighted_returns"].sum().to_frame("Strategy Returns")
    portfolio_df = pd.concat([portfolio_df,temp_df], axis = 0)

print(portfolio_df)

print(weights.stack().to_frame("weights").reset_index(level=0,drop=True))


############################# Visualize the Portfolio returns and compare it to that of the S&P 500 returns ##################################################


#Downloading the S&P 500 returns data till the latest date
spy = yf.download(tickers = "SPY",start = "2016-01-01",end = dt.date.today())
spy_return = np.log(spy[["Adj Close"]]).diff().dropna().rename({"Adj Close": "S&P500_returns"},axis = 1)
#Merging the S&P 500 returns with our portfolio dataframe
portfolio_df = portfolio_df.merge(spy_return,left_index = True,right_index = True)

#showcase the comparison of the strategy returns data yeild percentage in comparison with that of the S&P 500 
plt.style.use("ggplot")
portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1
portfolio_cumulative_return[:"2024-01-30"].plot(figsize = (16,6))
plt.title("Unsupervised Learning for Trading Strategy")
plt.ylabel("Returns")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.show()
