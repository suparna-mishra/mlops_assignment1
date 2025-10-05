# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, train_model, evaluate_model, preprocess

def main():
    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df,0.2,42)

    # Initialize model
    dtree = DecisionTreeRegressor(random_state=42)

    # Train model
    dtree = train_model(dtree, X_train, y_train)

    # Evaluate
    mse = evaluate_model(dtree, X_test, y_test)

    print(f"DecisionTreeRegressor Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
