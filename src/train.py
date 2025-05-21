import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from .pipeline import build_pipeline

def train(X, y, param_grid, experiment_name):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model = RandomForestClassifier()
        pipeline = build_pipeline(model)

        grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid.fit(X, y)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_score", grid.best_score_)
        mlflow.sklearn.log_model(grid.best_estimator_, "model")

        print("Best Params:", grid.best_params_)
        print("Best Score:", grid.best_score_)