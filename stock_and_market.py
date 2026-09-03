import numpy as np
import dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import os
import scipy as sc
import pandas as pd

dotenv.load_dotenv()

class StockData:

    def __init__(self, api_key=None, secret_key=None):
        self.api_key = api_key or os.environ["ALPACA_API_KEY"]
        self.secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]

        self.client = StockHistoricalDataClient(
            self.api_key,
            self.secret_key
        )

    def get_hourly_data(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.Series:

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=start,
            end=end,
        )

        response = self.client.get_stock_bars(request)

        df = response.df

        if df.empty:
            return pd.Series(dtype=float)

        # Keep closes indexed by timestamp for reliable alignment.
        prices = df["close"]
        if isinstance(prices.index, pd.MultiIndex):
            prices.index = prices.index.get_level_values(-1)
        prices = prices.sort_index()

        return prices

    def align_price_series(self, x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        aligned = pd.concat([x, y], axis=1, join="inner").dropna()
        if aligned.empty:
            return np.array([]), np.array([])
        return aligned.iloc[:, 0].to_numpy(), aligned.iloc[:, 1].to_numpy()

    def calculate_entropy(self, prices: np.ndarray) -> float:
        # Calculate the probability distribution of price changes
        price_changes = np.diff(prices)
        hist, bin_edges = np.histogram(price_changes, bins='auto', density=True)
        hist = hist[hist > 0]  # Remove zero probabilities

        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist))
        return entropy

    def calculate_kullback_leibler_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        if len(p) == 0 or len(q) == 0 or len(p) != len(q):
            return np.nan

        # Ensure both distributions are normalized
        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum == 0 or q_sum == 0:
            return np.nan

        p = p / p_sum
        q = q / q_sum
        q = np.clip(q, 1e-12, None)

        # Calculate Kullback-Leibler divergence
        kl_divergence = np.sum(np.where(p != 0, p * np.log(p / q), 0))
        return kl_divergence

    def calculate_corelation_coefficient(self, x: np.ndarray, y: np.ndarray) -> float: #do not use for short arrays
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return np.nan
        # Calculate the correlation coefficient
        x = np.diff(x)
        y = np.diff(y)
        if len(x) < 2 or len(y) < 2:
            return np.nan
        x = sc.ndimage.gaussian_filter1d(x, sigma=2)
        y = sc.ndimage.gaussian_filter1d(y, sigma=2)
        correlation_matrix = np.corrcoef(x, y)
        correlation_coefficient = correlation_matrix[0, 1]
        return correlation_coefficient

    def calculate_mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        if len(x) == 0 or len(y) == 0 or len(x) != len(y):
            return np.nan

        # Calculate the joint histogram
        joint_hist, _, _ = np.histogram2d(x, y, bins=bins)

        # Normalize the joint histogram to get the joint probability distribution
        joint_prob = joint_hist / np.sum(joint_hist)

        # Calculate the marginal probability distributions
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)

        # Calculate mutual information
        mutual_info = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint_prob[i, j] > 0:
                    mutual_info += joint_prob[i, j] * np.log(joint_prob[i, j] / (p_x[i] * p_y[j]))

        return mutual_info

    def calculate_dynamic_time_warping_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0 or len(y) == 0:
            return np.nan

        n, m = len(x), len(y)
        dtw_matrix = np.zeros((n + 1, m + 1))
        dtw_matrix[0, 1:] = np.inf
        dtw_matrix[1:, 0] = np.inf

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                              dtw_matrix[i, j - 1],    # deletion
                                              dtw_matrix[i - 1, j - 1]) # match

        return dtw_matrix[n, m]

class MarketReferences:
    #made for comparing stocks to market reference SNP500

    def __init__(self, api_key=None, secret_key=None):
        self.api_key = api_key or os.environ["ALPACA_API_KEY"]
        self.secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]

        self.client = StockHistoricalDataClient(
            self.api_key,
            self.secret_key
        )

    def compare_to_market(self, symbol: str, start: str, end: str) -> float:
        stock_data = StockData(self.api_key, self.secret_key)
        stock_prices = stock_data.get_hourly_data(symbol, start, end)
        market_prices = stock_data.get_hourly_data("SPY", start, end)  # Using SPY as a proxy for S&P 500
        stock_prices, market_prices = stock_data.align_price_series(stock_prices, market_prices)

        correlation_coefficient = stock_data.calculate_corelation_coefficient(stock_prices, market_prices)
        kullback_leibler_divergence = stock_data.calculate_kullback_leibler_divergence(stock_prices, market_prices)
        mutual_info = stock_data.calculate_mutual_information(stock_prices, market_prices)
        dtw_distance = stock_data.calculate_dynamic_time_warping_distance(stock_prices, market_prices)
        return correlation_coefficient, kullback_leibler_divergence, mutual_info, dtw_distance

# Example usage:
'''if __name__ == "__main__":
    print("Comparing AAPL to S&P 500 (SPY) from 2023-01-01 to 2023-12-31")
    market_ref = MarketReferences()
    correlation, kl_divergence, mutual_info, dtw_distance = market_ref.compare_to_market("AAPL", "2023-01-01", "2023-12-31")
    print(f"Correlation Coefficient: {correlation}")
    print(f"Kullback-Leibler Divergence: {kl_divergence}")
    print(f"Mutual Information: {mutual_info}")
    print(f"Dynamic Time Warping Distance: {dtw_distance}")
'''