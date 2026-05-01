import pandas as pd
import datetime as dt
import urllib.request, json
import os

from stock_market_lstm.config import DATA_SOURCE, TICKER, ALPHAVANTAGE_API_KEY


def load_data():

    if DATA_SOURCE == 'alphavantage':

        url_string = (
            "https://www.alphavantage.co/query"
            "?function=TIME_SERIES_DAILY"
            f"&symbol={TICKER}"
            "&outputsize=compact"
            f"&apikey={ALPHAVANTAGE_API_KEY}"
        )

        file_to_save = os.path.join(
            "data",
            "raw",
            f"stock_market_data-{TICKER}.csv"
        )

        os.makedirs(os.path.dirname(file_to_save), exist_ok=True)

        if not os.path.exists(file_to_save):

            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                print(data)
                if "Time Series (Daily)" not in data:
                    raise ValueError(f"AlphaVantage error: {data}")

                data = data["Time Series (Daily)"]

                df = pd.DataFrame(columns=['Date','Open','High','Low','Close'])

                for k, v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    df.loc[len(df)] = [
                        date.date(),
                        float(v['1. open']),
                        float(v['2. high']),
                        float(v['3. low']),
                        float(v['4. close'])
                    ]

            df.to_csv(file_to_save, index=False)
            print(f'Данные сохранены: {file_to_save}')

        else:
            print('Файл уже существует, загружаем CSV')
            df = pd.read_csv(file_to_save)

        return df

    else:
        raise ValueError("DATA_SOURCE должен быть 'alphavantage'")

