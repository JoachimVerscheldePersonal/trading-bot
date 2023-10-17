from datetime import datetime
from gym_trading_env.downloader import download
from ta.trend import IchimokuIndicator
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class MarketDataProvider:
    def __init__(self, exchange_name: str, symbol: str, timeframe: str, data_directory: str, since: datetime) -> None:
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_directory = data_directory
        self.since = since
        self.filename = f'{data_directory}/{exchange_name}-{symbol.replace("/","")}-{timeframe}.pkl'

    def download_data(self):
        # download 1h timeframe btcusd historical data starting from 2020 Jan 1
        download(exchange_names = [self.exchange_name],
            symbols= [self.symbol],
            timeframe= self.timeframe,
            dir = self.data_directory,
            since = self.since,
        )

    def fetch_data(self, include_ocean_theory_features: bool = True, include_ichimoku_features = True):
        self.download_data()
        features = ['feature_close',
                    'feature_open',
                    'feature_high',
                    'feature_low',
                    'feature_volume',
                    'close',
                    'open',
                    'high',
                    'low',
                    'volume']
        
        self.df = pd.read_pickle(self.filename)

        self.df['feature_open'] = self.df.open
        self.df['feature_high'] = self.df.high
        self.df['feature_low'] = self.df.low
        self.df['feature_close'] = self.df.close
        self.df['feature_volume'] = self.df.volume
        
        if include_ichimoku_features:
            ichimoku_indicator = IchimokuIndicator(high=self.df.feature_high, low=self.df.feature_low)
            features.extend(['feature_senkou_a', 'feature_senkou_b', 'feature_kijun', 'feature_tenkan'])
            self.df['feature_senkou_a'] = ichimoku_indicator.ichimoku_a()
            self.df['feature_senkou_b'] = ichimoku_indicator.ichimoku_b()
            self.df['feature_kijun'] = ichimoku_indicator.ichimoku_base_line()
            self.df['feature_tenkan'] = ichimoku_indicator.ichimoku_conversion_line()

        if include_ocean_theory_features:
            ocean_theory_indicator = OceanTheoryIndicator(10, 'feature_close')
            features.extend(['feature_natural_market_river',
                             'feature_natural_market_mirror',
                             'feature_natural_market_mirror_nma', 
                             'feature_natural_market_river_nma',
                             'feature_natural_market_mirror_nma_diff',
                             'feature_natural_market_river_nma_diff'
                             ])
            
            self.df['feature_natural_market_river'] = ocean_theory_indicator.natural_market_river(self.df,'feature_close', 10).values
            self.df['feature_natural_market_mirror'] = ocean_theory_indicator.natural_market_mirror(self.df, 'feature_close', 10).values

            self.df['feature_natural_market_mirror_nma'] = ocean_theory_indicator.natural_moving_average(self.df, 'feature_natural_market_mirror',20, True).values
            self.df['feature_natural_market_river_nma'] = ocean_theory_indicator.natural_moving_average(self.df, 'feature_natural_market_river',20, True).values

            self.df['feature_natural_market_mirror_nma_diff'] = self.df.feature_natural_market_mirror_nma.diff().values
            self.df['feature_natural_market_river_nma_diff'] =  self.df.feature_natural_market_river_nma.diff().values 

            scaler = StandardScaler()
            self.df[features] = scaler.fit_transform(self.df[features])
            self.df.dropna(inplace=True)
            return self.df


class OceanTheoryIndicator():
    
    def __init__(self, horizon: int, close_price_property: str) -> None:
        self.horizon = horizon
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

