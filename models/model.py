#model.py
import lightgbm as lgb

def get_model(objective="quantile", alpha=0.1, random_state=42):
    return lgb.LGBMRegressor(
        objective=objective,
        alpha=alpha,
        n_estimators=8000,          # let early stopping decide
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,       # more conservative
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1
    )
    