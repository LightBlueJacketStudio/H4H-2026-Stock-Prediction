# evaluate.py
import numpy as np

def widen_interval(p_low, p_high, k):
    mid = 0.5 * (p_low + p_high)
    half = 0.5 * (p_high - p_low)
    return mid - k * half, mid + k * half

def find_k_for_target_coverage(p_low, p_high, true_close, target=0.80):
    ks = np.linspace(1.0, 3.0, 201)
    best_k, best_cov = None, None
    for k in ks:
        lo, hi = widen_interval(p_low, p_high, k)
        cov = ((true_close >= lo) & (true_close <= hi)).mean()
        if best_k is None or abs(cov - target) < abs(best_cov - target):
            best_k, best_cov = k, cov
    return best_k, best_cov

def evaluate_interval(models, X_test, y_test, close_test):
    """
    models: (m10, m90) trained on y_ret (log return)
    y_test: realized log return on test set
    close_test: Close_t for test rows (today's close)
    """
    m10, m90 = models

    r10 = m10.predict(X_test)
    r90 = m90.predict(X_test)

    # ensure proper ordering even if models cross
    r_low = np.minimum(r10, r90)
    r_high = np.maximum(r10, r90)

    close = close_test.values
    y_true = y_test.values

    # Convert log-return interval -> price interval
    p_low  = close * np.exp(r_low)
    p_high = close * np.exp(r_high)

    # True tomorrow close
    true_close = close * np.exp(y_true)

    # ---------------------------------
    # CALIBRATION STEP (optional)
    # ---------------------------------
    #k, achieved_cov = find_k_for_target_coverage(
    #    p_low, p_high, true_close, target=0.80
    #)

    #p_low, p_high = widen_interval(p_low, p_high, k)

    # print("Calibrated k:", k)
    # print("Coverage after calibration:", achieved_cov)

    # Coverage
    covered = (true_close >= p_low) & (true_close <= p_high)
    coverage = covered.mean()

    # Avg dollar width
    width = np.mean(p_high - p_low)

    # Miss penalty
    below = np.maximum(p_low - true_close, 0.0)
    above = np.maximum(true_close - p_high, 0.0)
    miss_penalty = np.mean(below + above)

    return coverage, width, miss_penalty