from metaflow import FlowSpec, step, Parameter
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

class WineScoringFlow(FlowSpec):
    model_name = Parameter('model_name', default='wine-classifier')
    model_stage = Parameter('model_stage', default='None')
    random_state = Parameter('random_state', default=42)
    
    @step
    def start(self):
        print("Loading data for scoring...")
        
        wine = datasets.load_wine()
        X = wine.data
        y = wine.target
        
        # Create a small subset to simulate new data
        _, X_new, _, y_new = train_test_split(
            X, y, test_size=0.1, random_state=self.random_state
        )
        
        self.X_new = X_new
        self.y_true = y_new
        self.target_names = wine.target_names
        
        print(f"Data loaded for scoring. Shape: {self.X_new.shape}")
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_new_scaled = scaler.fit_transform(self.X_new)
        
        print("Data preprocessing complete")
        self.next(self.load_model)
    
    @step
    def load_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        try:
            if self.model_stage == 'None':
                self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/latest")
            else:
                self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{self.model_stage}")
            print("Model loaded from registry")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        self.next(self.score_data)
    
    @step
    def score_data(self):
        self.predictions = self.model.predict(self.X_new_scaled)
        self.prediction_probas = self.model.predict_proba(self.X_new_scaled)
        
        prediction_df = pd.DataFrame({
            'true_label': [self.target_names[i] for i in self.y_true],
            'predicted_label': [self.target_names[i] for i in self.predictions],
        })
        
        for i, class_name in enumerate(self.target_names):
            prediction_df[f'prob_{class_name}'] = self.prediction_probas[:, i]
        
        self.prediction_df = prediction_df
        self.accuracy = (self.predictions == self.y_true).mean()
        
        print(f"Accuracy on new data: {self.accuracy:.4f}")
        self.next(self.end)
    
    @step
    def end(self):
        print(f"Accuracy: {self.accuracy:.4f}")
        print("\nSample predictions:")
        print(self.prediction_df.head())

if __name__ == "__main__":
    WineScoringFlow()