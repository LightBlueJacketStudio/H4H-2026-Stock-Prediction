Steps to Run:

1. set up and activate virtual Environment
must use python 3.11
```
rmdir -r .venv
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install yfinance pandas numpy scikit-learn
```

2. Install required packages
```
pip install -r requirements.txt
```

3. Prepare a FinnHub API key
```
create a file called ".env"
put the API key under .env as <FINHUB_API_KEY="YOUR_API_KEY">
```

4. Run Aggregator.py to test the connection of everything
```
python aggregator.py
```

# Glimpse – UI (Streamlit Frontend)
1. You need to install

```
pip install streamlit
pip install protobuf==3.20.3
pip install altair
```
2. run stream_lit app
```
streamlit run streamlit_app/app.py
