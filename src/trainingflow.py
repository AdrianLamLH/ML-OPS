from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class WineTrainingFlow(FlowSpec):
    random_state = Parameter('random_state', default=42)
    test_size = Parameter('test_size', default=0.2)
    cv_folds = Parameter('cv_folds', default=5)

    @step
    def start(self):
        wine = datasets.load_wine()
        self.X = wine.data
        self.y = wine.target
        self.feature_names = wine.feature_names
        self.target_names = wine.target_names
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Data loaded. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scaler = scaler
        self.next(self.train_models)

    @step
    def train_models(self):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            rf, param_grid, cv=self.cv_folds, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_
        
        print(f"Best parameters: {self.best_params}")
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        self.y_pred = self.best_model.predict(self.X_test_scaled)
        
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.classification_report = classification_report(
            self.y_test, self.y_pred, target_names=self.target_names
        )
        
        print(f"Accuracy: {self.accuracy:.4f}")
        print(self.classification_report)
        self.next(self.register_model)

    @step
    def register_model(self):
        # Set MLFlow tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("wine-classification")
        
        with mlflow.start_run() as run:
            mlflow.log_params(self.best_params)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_metric("accuracy", self.accuracy)
            
            mlflow.sklearn.log_model(
                self.best_model, 
                "model",
                registered_model_name="wine-classifier"
            )
            
            self.run_id = run.info.run_id
        
        print(f"Model registered. Run ID: {self.run_id}")
        self.next(self.end)

    @step
    def end(self):
        print(f"Best model parameters: {self.best_params}")
        print(f"Test accuracy: {self.accuracy:.4f}")
        print(f"MLFlow run ID: {self.run_id}")

if __name__ == "__main__":
    WineTrainingFlow()