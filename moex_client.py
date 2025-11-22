import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Optional


class MoexClient:
    BASE_URL = "https://iss.moex.com/iss"
    DEFAULT_ENGINE = "stock"
    DEFAULT_MARKET = "shares"
    DEFAULT_BOARD = "TQBR"

    __slots__ = ("ticker", "engine", "market", "board")

    def __init__(self, ticker: str, engine: str = DEFAULT_ENGINE, market: str = DEFAULT_MARKET, board: str = DEFAULT_BOARD):
        self.ticker = ticker.upper()
        self.engine = engine
        self.market = market
        self.board = board

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.BASE_URL}/{endpoint}"
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _interval_to_code(interval: str) -> int:
        mapping = {"1m": 1, "10m": 10, "1h": 60}
        return mapping.get(interval, 24 * 60)

    def get_history(self, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        if interval not in {"1d", "1h", "10m", "1m"}:
            raise ValueError("interval must be: '1d', '1h', '10m', '1m'")

        if interval == "1d":
            endpoint = f"history/engines/{self.engine}/markets/{self.market}/boards/{self.board}/securities/{self.ticker}.json"
            params = {"from": start, "till": end}
        else:
            endpoint = f"engines/{self.engine}/markets/{self.market}/securities/{self.ticker}/candles.json"
            params = {"from": start, "till": end, "interval": self._interval_to_code(interval)}

        all_dfs = []
        start_index = 0
        limit = 100

        while True:
            params["start"] = start_index
            params["limit"] = limit
            j = self._get(endpoint, params)
            block = j.get("history") or j.get("candles")
            if not block:
                break

            data, cols = block.get("data", []), block.get("columns", [])
            if not data:
                break
            df = pd.DataFrame(data, columns=cols)
            if "end" in df:
                df["TRADEDATE"] = pd.to_datetime(df["end"])
                df.rename(columns={"close": "CLOSE"}, inplace=True)
            else:
                df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])

            if "CLOSE" not in df:
                break

            volume_col = "VOLUME" if "VOLUME" in df else None
            if volume_col:
                all_dfs.append(df[["TRADEDATE", "CLOSE", volume_col]])
            else:
                all_dfs.append(df[["TRADEDATE", "CLOSE"]])

            if len(df) < limit:
                break
            start_index += limit

        if not all_dfs:
            raise ValueError(f"No data for {self.ticker} for period {start} — {end}")

        df = pd.concat(all_dfs, ignore_index=True)
        df.drop_duplicates(subset="TRADEDATE", inplace=True)
        df.sort_values("TRADEDATE", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={"TRADEDATE": "TIMESTAMP", "CLOSE": "PRICE"}, inplace=True)
        df.rename(columns={"volume": "VOLUME"}, inplace=True)
        print(f"Fetch {len(df)} lines {self.ticker} ({interval}): {start} -> {end}")
        return df


    def get_data(self) -> Dict:
        endpoint = f"engines/{self.engine}/markets/{self.market}/securities/{self.ticker}.json"
        j = self._get(endpoint)
        data = j.get("marketdata", {}).get("data", [])
        cols = j.get("marketdata", {}).get("columns", [])

        if not data:
            raise ValueError(f"No data for {self.ticker}")

        df = pd.DataFrame(data, columns=cols)
        price = None
        volume = None
        for field in ["LCURRENTPRICE", "LAST", "MARKETPRICE", "LASTPRICE", "CLOSEPRICE"]:
            if field in df and pd.notna(df[field].iloc[0]):
                price = df[field].iloc[0]
                break
        else:
            price = float("nan")

        if "VOLUME" in df:
            volume = df["VOLUME"].iloc[0]
        else:
            volume = float("nan")

        time = df["SYSTIME"].iloc[0] if "SYSTIME" in df else datetime.now().isoformat()

        return {
            "ticker": self.ticker,
            "price": price,
            "volume": volume,
            "time": time
    }


    def get_securities_list(self, market: str = "shares") -> pd.DataFrame:
        endpoint = f"engines/{self.engine}/markets/{market}/securities.json"
        j = self._get(endpoint)
        data = j.get("securities", {}).get("data", [])
        cols = j.get("securities", {}).get("columns", [])
        return pd.DataFrame(data, columns=cols)

    def process(self, history: pd.DataFrame) -> pd.DataFrame:
        if "PRICE" not in history:
            raise ValueError("DataFrame does not contain certain columns")

        df = history.copy()
        df["LOGRET"] = np.log(df["PRICE"] / df["PRICE"].shift(1))
        df["RET"] = df["PRICE"].pct_change()
        return df

    def plot(self, history: pd.DataFrame):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=500, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(history.index, history['PRICE'], color='black', linewidth=1.5, label='Цена')
        ax1.set_ylabel('Цена', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.bar(history.index, history['VOLUME'], color='blue', alpha=0.7, label='Объем')
        ax2.set_ylabel('Объем', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax2.set_xlabel('Дата/Время', fontsize=12)
        fig.suptitle('Цена и объем', fontsize=16)

        plt.tight_layout()
        plt.show()


