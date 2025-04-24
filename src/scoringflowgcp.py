from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, resources, retry, timeout, catch
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

@conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.8.0', 'databricks-cli':'0.17.0'}, python='3.9.16')
class WineScoringFlowGCP(FlowSpec):
    model_name = Parameter('model_name', default='WineClassifierGCP')
    model_stage = Parameter('model_stage', default='None')
    random_state = Parameter('random_state', default=42)
    test_size = Parameter('test_size', default=0.2)

    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        
        # Load the wine dataset
        wine = datasets.load_wine()
        self.X = wine.data
        self.y = wine.target
        self.feature_names = wine.feature_names
        self.target_names = wine.target_names
        
        # Use same split as training flow
        _, self.X_test, _, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Test data loaded. Shape: {self.X_test.shape}")
        self.next(self.load_model)

    @retry(times=3)
    @step
    def load_model(self):
        mlflow.set_tracking_uri("http://mlflow.default.svc.cluster.local:5000")
        
        # Get model from registry
        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # Try to get scaler or create fallback
        try:
            self.scaler = mlflow.sklearn.load_model(f"models:/WineScaler/{self.model_stage}")
            print("Found scaler in model registry")
        except Exception as e:
            # Fallback if scaler not in registry
            print(f"No scaler found: {str(e)}")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        
        print(f"Loaded model '{self.model_name}' from '{self.model_stage}' stage")
        self.next(self.score_model)

    @kubernetes
    @resources(cpu=1, memory=2000)
    @timeout(minutes=5)
    @catch(var='caught_exception', print_exception=True)
    @step
    def score_model(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_test)  # Fit it first
        X_test_scaled = self.scaler.transform(self.X_test)        
        # Run scoring
        with mlflow.start_run(run_name="wine-scoring-kubernetes") as run:
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(self.y_test, y_pred)
            
            # Log results to MLflow
            mlflow.log_metric("test_accuracy", self.accuracy)
            
            # Also log some sample predictions
            sample_size = min(5, len(y_pred))
            for i in range(sample_size):
                mlflow.log_metric(f"sample_pred_{i}", y_pred[i])
            
            print(f"Test accuracy: {self.accuracy:.4f}")
            
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow completed successfully!")
        print(f"Test accuracy: {self.accuracy:.4f}")
        if hasattr(self, 'accuracy'):
            print(f"Test accuracy: {self.accuracy:.4f}")
        else:
            print("No accuracy score available")

if __name__ == '__main__':
    WineScoringFlowGCP()