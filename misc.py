import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data():
    """
    Downloads the Boston housing dataset from CMU
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    # Reconstruct dataset
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df


def preprocess(df, test_size: float = 0.2, random_state: int = 42):
    """splits into train/test.
    and returns X_train, X_test, y_train, y_test"""
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
	
	

def train_model(model, X_train, y_train):
    """
    Fits any scikit-learn style model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model on test data and returns MSE.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse
