import yfinance as yf

import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
#database_url = os.getenv("DATABASE_URL")
FINNHUB_API = os.getenv("FINHUB_API_KEY")

ticker = "AMZN"
df = yf.download(ticker, period="5y", interval="1d")

print(df.head())

import finnhub
from datetime import date, timedelta

client = finnhub.Client(api_key=FINNHUB_API)

symbol = "AMZN"
to_ = date.today().isoformat()
from_ = (date.today() - timedelta(days=7)).isoformat()

news = client.company_news(symbol, _from=from_, to=to_)  # _from is required name
# news is a list of dicts: headline, source, url, datetime, summary, etc.
print(news[:3])