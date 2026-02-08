import pandas as pd

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values

    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values

    return X_train, y_train, X_test, y_test
