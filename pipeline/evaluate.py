import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    direction = (np.sign(preds) == np.sign(y_test)).mean()
    return rmse, direction