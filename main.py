import yaml
from src.data import load_data, clean_data
from src.train import train

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    df = load_data(config["data"]["path"])
    df = clean_data(df)

    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    param_grid = config["model"]["param_grid"]
    experiment_name = config["experiment_name"]

    train(X, y, param_grid, experiment_name)