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


use 
```
python -m test.test_feature
```
to see the intermediate result
-m: always treat project root as working dir
tes.test_feature: module calling
__init__.py is necessary for a folder to be treated as module, then we can use import from that folder