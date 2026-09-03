import os
from datetime import datetime, date
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionTradesRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType, AssetStatus

load_dotenv()

# Define explicit DataFrame columns to prevent KeyErrors on empty responses
CONTRACT_COLUMNS = [
    "option_symbol",
    "underlying",
    "expiration",
    "strike",
    "type",
    "style",
    "contract_size",
    "status",
    "tradable",
]


class OptionsData:

    def __init__(self, api_key=None, secret_key=None):
        self.api_key = api_key or os.environ["ALPACA_API_KEY"]
        self.secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]

        self.data_client = OptionHistoricalDataClient(
            self.api_key, self.secret_key
        )

        self.trading_client = TradingClient(
            self.api_key, self.secret_key, paper=True
        )

    def get_contracts(
        self,
        underlying: str,
        expiration_start: str | date,
        expiration_end: str | date,
        option_type: str | ContractType | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        status: str | AssetStatus | None = None,
    ) -> pd.DataFrame:

        if isinstance(expiration_start, str):
            expiration_start = date.fromisoformat(expiration_start)
        if isinstance(expiration_end, str):
            expiration_end = date.fromisoformat(expiration_end)

        if isinstance(option_type, str):
            option_type = ContractType(option_type.lower())

        if isinstance(status, str):
            status = AssetStatus(status.lower())

        all_contracts = []
        page_token = None

        while True:
            request = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date_gte=expiration_start,
                expiration_date_lte=expiration_end,
                type=option_type,
                status=status,
                strike_price_gte=(
                    str(strike_min) if strike_min is not None else None
                ),
                strike_price_lte=(
                    str(strike_max) if strike_max is not None else None
                ),
                limit=10000,
                page_token=page_token,
            )

            response = self.trading_client.get_option_contracts(request)

            if response.option_contracts:
                all_contracts.extend(response.option_contracts)

            page_token = response.next_page_token

            if not page_token:
                break

        if not all_contracts:
            return pd.DataFrame(columns=CONTRACT_COLUMNS)

        rows = []
        for contract in all_contracts:
            rows.append({
                "option_symbol": contract.symbol,
                "underlying": contract.underlying_symbol,
                "expiration": contract.expiration_date,
                "strike": float(contract.strike_price),
                "type": contract.type,
                "style": contract.style,
                "contract_size": int(contract.size),
                "status": contract.status,
                "tradable": contract.tradable,
            })

        return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)

    def get_trades(
        self,
        underlying: str,
        start: str | datetime,
        end: str | datetime,
        expiration_start: str | date,
        expiration_end: str | date,
        option_type: str | ContractType | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        status: str | AssetStatus | None = None,
    ) -> pd.DataFrame:

        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        contracts = self.get_contracts(
            underlying=underlying,
            expiration_start=expiration_start,
            expiration_end=expiration_end,
            option_type=option_type,
            strike_min=strike_min,
            strike_max=strike_max,
            status=status,
        )

        if contracts.empty:
            return pd.DataFrame()

        symbols = contracts["option_symbol"].tolist()
        all_trades = []

        for i in range(0, len(symbols), 100):
            symbol_batch = symbols[i : i + 100]
            page_token = None

            while True:
                request = OptionTradesRequest(
                    symbol_or_symbols=symbol_batch,
                    start=start,
                    end=end,
                    limit=1000,
                    page_token=page_token,
                )

                response = self.data_client.get_option_trades(request)
                df = response.df

                if not df.empty:
                    all_trades.append(df)

                page_token = getattr(response, "next_page_token", None)

                if not page_token:
                    break

        if not all_trades:
            return pd.DataFrame()

        trades = pd.concat(all_trades).reset_index()

        trades = trades.merge(
            contracts,
            left_on="symbol",
            right_on="option_symbol",
            how="left",
        )

        if "timestamp" in trades.columns:
            trades = trades.sort_values(["timestamp", "symbol"]).reset_index(
                drop=True
            )

        return trades


# Example usage:
'''if __name__ == "__main__":
    options = OptionsData()

    # Pass status="inactive" to pull expired contracts from past dates
    contracts = options.get_contracts(
        underlying="AAPL",
        expiration_start="2025-01-17",
        expiration_end="2025-04-30",
        status="inactive",
    )

    print("Number of contracts:", len(contracts))

    if not contracts.empty:
        print(contracts.head(20))
        print("Earliest Expiration:", contracts["expiration"].min())
        print("Latest Expiration:", contracts["expiration"].max())
    else:
        print("No contracts found matching criteria.")'''