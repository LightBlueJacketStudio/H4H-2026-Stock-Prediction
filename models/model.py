import lightgbm as lgb

def get_model():
    return lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )