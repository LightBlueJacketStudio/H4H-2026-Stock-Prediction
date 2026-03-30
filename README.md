Setting up Environments:

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


Prepare the backend Data:
1. Change Your Directory to Project Root
```
cd <path to H4H-2026-Stock-Prediction>
```
2. Get Price Data
```
python .\pipeline\OHLCV_aggregator.py
```

3. Get News Data
```
python .\pipeline\news_aggregator.py
```

4. Inference:
```
python -m pipeline.main
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
```


use 
```
python -m test.test_feature
```
to see the intermediate result
-m: always treat project root as working dir
tes.test_feature: module calling
__init__.py is necessary for a folder to be treated as module, then we can use import from that folder