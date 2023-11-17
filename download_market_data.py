from data_providers import MarketDataProvider
from datetime import datetime
symbols = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "BCC/USDT",
    "NEO/USDT",
    "LTC/USDT",
    "QTUM/USDT",
    "ADA/USDT",
    "XRP/USDT",
    "EOS/USDT",
    "TUSD/USDT",
    "IOTA/USDT",
    "XLM/USDT",
    "ONT/USDT",
    "TRX/USDT",
    "ETC/USDT",
    "ICX/USDT",
    "VEN/USDT",
    "NULS/USDT",
    "VET/USDT",
]

data_provider = MarketDataProvider(exchange_name="binance", symbols=symbols, timeframe="1h", data_directory="data/test", since=datetime(year= 2022, month= 1, day=1), until=datetime(year= 2022, month= 12, day=31))
data_provider.fetch_data()
