from datetime import datetime
from gym_trading_env.downloader import download
from ta.momentum import stochrsi, ultimate_oscillator
from ta.trend import macd
from ta.volatility import average_true_range
import pandas as pd
import numpy as np
import os.path
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class PreprocessingPipeline:

    def fit_transform(self, X: pd.DataFrame, skip_timezone_conversion:bool = False):
        ocean_indicator = OceanTheoryIndicator(close_price_property="close")
        X['feature_high'] = X.high
        X['feature_low'] = X.low
        X['feature_close'] = X.close
        X['feature_volume'] = X.volume
        X['feature_returns'] = X.close.pct_change()

        X['feature_ret_5'] = X.close.pct_change(4)
        X['feature_ret_10'] = X.close.pct_change(8)
        X['feature_ret_12'] = X.close.pct_change(12)
        X['feature_ret_24'] = X.close.pct_change(24)
        X['feature_rsi'] = stochrsi(X.close)
        X['feature_macd'] = macd(X.close)
        X['feature_atr'] = average_true_range(X.high, X.low, X.close)

        if not skip_timezone_conversion:
            X['date_close'] = X['date_close'].dt.tz_convert(None)

        X['feature_ultosc'] = ultimate_oscillator(X.high, X.low, X.close)
        X["feature_nmr_24"] = ocean_indicator.natural_market_river(X, "close", 24).values

        non_scalable_features = ["open","high","low","close","date_open", "date_close"]
        features_to_scale = [col for col in X.columns if col not in non_scalable_features]
        
        scaler = MinMaxScaler()
        X[features_to_scale] = scaler.fit_transform(X[features_to_scale])
        
        X.dropna(inplace=True)

        return X
      
        
class YahooMarketDataProvider:
    def __init__(self):
        pass

    def download_data(self, ticker, timeframe, from_date, to_date):
        # Download historical data using yfinance
        data = yf.download(
            ticker,
            start=from_date,
            end=to_date,
            interval=timeframe,
            progress=False
        )
        # Extract relevant features (timestamp, open, high, low, close, volume)
        if not data.empty:
            data = data[['High', 'Low', 'Close', 'Volume']]
            data.reset_index(inplace=True)
            data.columns = ['date_close', 'high', 'low', 'close', 'volume']
        else:
            print(f"No data available for {ticker} in the specified date range.")
            data = pd.DataFrame(columns=['date_close', 'high', 'low', 'close', 'volume'])

        return data

    def get_data(self, ticker, timeframe, from_date, to_date):
        df = self.download_data(ticker, timeframe, from_date, to_date)
        preproc_pipeline = PreprocessingPipeline()
        df = preproc_pipeline.fit_transform(df, True)
        return df
    
class MarketDataProvider:
    def __init__(self, exchange_name: str, symbols: list[str], timeframe: str, data_directory: str, since: datetime, until: datetime) -> None:
        self.exchange_name = exchange_name
        self.symbols = symbols
        self.timeframe = timeframe
        self.data_directory = data_directory
        self.since = since
        self.until = until

    def get_file_name(self, symbol: str):
        return f'{self.exchange_name}-{symbol.replace("/","")}-{self.timeframe}.pkl'

    def get_raw_file_path(self, symbol: str):
        return os.path.join(self.data_directory,self.get_file_name(symbol))
    
    def get_preprocessed_file_path(self, symbol: str):
        return os.path.join(self.data_directory, "preprocessed", self.get_file_name(f'PREPROC_{symbol}'))
    

    def download_data(self):
        # download only when files do not exist yet
        symbols_to_download = [symbol for symbol in self.symbols if not os.path.exists(self.get_raw_file_path(symbol))]
        if len(symbols_to_download) > 0:
            download(exchange_names = [self.exchange_name],
                symbols= symbols_to_download,
                timeframe= self.timeframe,
                dir = self.data_directory,
                since = self.since,
                until = self.until
            )

    def fetch_data(self):
        self.download_data()
        for symbol in self.symbols:
            # skip if already preprocessed
            preprocessed_file_name = self.get_preprocessed_file_path(symbol)
            if os.path.exists(preprocessed_file_name):
                continue

            df = pd.read_pickle(self.get_raw_file_path(symbol))
            if len(df) == 0:
                continue
            
            preproc_pipeline = PreprocessingPipeline()
            df = preproc_pipeline.fit_transform(df,True)

            print(f'{preprocessed_file_name}: post drop size {len(df)}')
            df.to_pickle(preprocessed_file_name)
    
class OceanTheoryIndicator():
    def __init__(self, close_price_property: str) -> None:
        self.close_price_property = close_price_property

    def add_epislon_to_zero(self, series: pd.Series):
        series[series.where(series == 0).index] += np.finfo(float).eps
        return series

    # Ocean Theory
    def ocean_index(self, df: pd.DataFrame, close_property, index_number, skip_log_difference = False):
        ocean_indices = pd.Series(np.zeros(len(df)))
        current_price = self.add_epislon_to_zero(df[close_property][index_number:])
        historical_price = self.add_epislon_to_zero(df.shift(index_number)[close_property][index_number:])
        
        if skip_log_difference:
            log_return = historical_price - current_price
        else:
            log_return =  np.log(historical_price) - np.log(current_price)

        ocean_indices.iloc[index_number:] = log_return / np.sqrt(index_number)
        return ocean_indices

    # Natural Market Mirror
    def natural_market_mirror(self, df:pd.DataFrame ,close_property, reachback):
        cumulative_indices = pd.Series(np.zeros(len(df)))
        nma = pd.Series(np.zeros(len(df)))
        
        for i in range(1, reachback+1):
            cumulative_indices += self.ocean_index(df,close_property,i)
        nma[reachback:] = cumulative_indices[reachback:] / reachback

        return nma

    # Natural Market River
    def natural_market_river(self,df: pd.DataFrame ,close_property, reachback, skip_log_difference = False):
        cumulative_indices = pd.Series(np.zeros(len(df)))
        nmr = pd.Series(np.zeros(len(df)))

        for i in range(1, reachback+1):
            cumulative_indices += (np.sqrt(i)-np.sqrt(i-1))*self.ocean_index(df,close_property,i, skip_log_difference)
        nmr[reachback:] = cumulative_indices[reachback:] / reachback

        return nmr

    def exponential_moving_average(self, signal:np.array, points:int, exponential_const: np.array):
        """
        Calculate the N-point exponential moving average of a signal

        Inputs:
            signal: numpy array -   A sequence of price points in time
            points:      int    -   The size of the moving average
            exponential_const: numpy array    -   The smoothing factor

        Outputs:
            ma:     numpy array -   The moving average at each point in the signal
        """
        ema = np.zeros(len(signal))
        ema[0] = signal[0]

        for i in range(1, len(signal)):
            ema[i] = (signal[i] * exponential_const[i]) + (ema[i - 1] * (1 - exponential_const[i]))

        return pd.Series(ema)

    def natural_moving_average(self, df:pd.DataFrame, price_property: str, periods:int, skip_log_difference = False):
        nma = pd.Series(np.zeros(len(df)))
        o1_over_periods = self.ocean_index(df,price_property, 1, skip_log_difference).abs().rolling(min_periods=periods, window=periods).sum()[periods:]
        natural_market_river_o1 = self.natural_market_river(df, price_property,1, skip_log_difference).abs()[periods:]
        exponential_constant = natural_market_river_o1.divide(o1_over_periods)
        nma[periods:] = self.exponential_moving_average(df[price_property].iloc[periods:].to_numpy(),periods, exponential_constant.to_numpy())
        return nma

    def smooth_function(self, df: pd.DataFrame, property_name, start_component):
        smoothed_function = pd.Series(np.zeros(len(df)))
        x = df.index
        y = df[property_name]
        rft = np.fft.rfft(y) # perform real fourier transform
        rft[start_component:] = 0   # When to start removing components
        y_smooth = pd.Series(np.fft.irfft(rft)) # perform inverse fourier
        print(y_smooth)
        smoothed_function[1:] = y_smooth
        
        return smoothed_function


       
