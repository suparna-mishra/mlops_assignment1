from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

from misc import load_data, preprocess

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf"))
    ])

    param_grid = {
        "krr__alpha": np.logspace(-3, 2, 10),
        "krr__gamma": np.logspace(-3, 1, 10),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Best params:", gs.best_params_)
    print(f"Kernel Ridge (RBF) Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()

