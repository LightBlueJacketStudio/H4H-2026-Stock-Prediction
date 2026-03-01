

from models import get_model

def train_model(df, split_date="2024-01-01"):
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train = X[X.index < split_date]
    X_test  = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test  = y[y.index >= split_date]

    model = get_model()
    model.fit(X_train, y_train)

    return model, X_test, y_test