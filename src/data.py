import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
    df = df.dropna(subset=["Embarked"])
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    return df