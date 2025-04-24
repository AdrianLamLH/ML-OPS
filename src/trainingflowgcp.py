from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, resources, retry, timeout, catch, schedule
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, classification_report

@conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.8.0', 'databricks-cli':'0.17.0'}, python='3.9.16')
class WineTrainingFlowGCP(FlowSpec):
    random_state = Parameter('random_state', default=42)
    test_size = Parameter('test_size', default=0.2)
    cv_folds = Parameter('cv_folds', default=5)

    @step
    def start(self):
        from sklearn import datasets
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

    @retry(times=3)
    @step
    def feature_engineering(self):
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scaler = scaler
        self.next(self.train_models)

    @kubernetes
    @resources(cpu=2, memory=4000)
    @timeout(minutes=10)
    @retry(times=2)
    @catch(var='caught_exception', print_exception=True)
    @step
    def train_models(self):
        mlflow.set_tracking_uri("http://mlflow.default.svc.cluster.local:5000")
        mlflow.set_experiment("Wine-GCP-Kubernetes")

        with mlflow.start_run(run_name="wine-rf-kubernetes") as run:
            # Log basic parameters
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("cv_folds", self.cv_folds)
            
            # Define RF hyperparameter search space
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            
            # Train model with grid search
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(rf, param_grid, cv=self.cv_folds, scoring='accuracy')
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Save best model and params
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            # Evaluate on test set
            y_pred = self.best_model.predict(self.X_test_scaled)
            self.accuracy = accuracy_score(self.y_test, y_pred)
            self.report = classification_report(self.y_test, y_pred, target_names=self.target_names)
            
            # Log performance metrics
            mlflow.log_metrics({
                "accuracy": self.accuracy,
                "best_score": grid_search.best_score_
            })
            
            # Log tuned hyperparameters
            for param, value in self.best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Save model artifacts
            mlflow.sklearn.log_model(self.best_model, "model")
            
            # Register in model registry for deployment
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "WineClassifierGCP")
            
            # Print results to console
            print(f"Best parameters: {self.best_params}")
            print(f"Test accuracy: {self.accuracy:.4f}")
            print(f"Classification report:\n{self.report}")
        
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed successfully!")
        print(f"Best model parameters: {self.best_params}")
        print(f"Test accuracy: {self.accuracy:.4f}")

if __name__ == '__main__':
    WineTrainingFlowGCP()