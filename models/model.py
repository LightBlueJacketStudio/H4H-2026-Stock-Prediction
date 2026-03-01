#model.py
import lightgbm as lgb

def get_model(objective="quantile", alpha=0.1, random_state=42):
    return lgb.LGBMRegressor(
        objective=objective,
        alpha=alpha,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1
    )
    