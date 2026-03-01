import numpy as np

def evaluate_interval(models, X_test, y_test, close_test):
    m10, m90 = models
    q10 = m10.predict(X_test)
    q90 = m90.predict(X_test)

    # convert predicted return quantiles → price bounds
    lower = close_test.values * (1 + q10)
    upper = close_test.values * (1 + q90)

    # true tomorrow close from y_test + today's close:
    true_close = close_test.values * (1 + y_test.values)

    covered = np.mean((true_close >= lower) & (true_close <= upper))
    avg_width = np.mean(upper - lower)

    # optional: interval score-ish metric (smaller better, penalize misses)
    miss_penalty = np.mean(np.maximum(lower - true_close, 0) + np.maximum(true_close - upper, 0))

    return covered, avg_width, miss_penalty