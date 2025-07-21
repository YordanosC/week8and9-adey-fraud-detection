import pandas as pd
import numpy as np
# Add these to your existing sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# Add these for neural networks
from keras.layers import SimpleRNN, Dense
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras  # Importing keras logging
import time

class ModelPipeline:
    """
    A class for building a machine learning pipeline for model selection, training, and evaluation.
    """

    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.performance_metrics = {}
        self.y_probs = {}
        
    def add_models(self):
        """Add models to the models dictionary."""
        # Classical Machine Learning Models
        self.models['Logistic Regression'] = LogisticRegression(max_iter=1000)
        self.models['Decision Tree'] = DecisionTreeClassifier()
        self.models['Random Forest'] = RandomForestClassifier()
        self.models['Gradient Boosting'] = GradientBoostingClassifier()

        # Neural Networks
        self.models['MLP'] = self.build_mlp_model()
        self.models['RNN'] = self.build_rnn_model()
        self.models['LSTM'] = self.build_lstm_model()
        self.models['CNN'] = self.build_cnn_model()

    def build_lstm_model(self):
        """
        Builds an LSTM model.
        """
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    def build_mlp_model(self):
        """Build a Multi-Layer Perceptron (MLP) model."""
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_rnn_model(self):
        """Build a Recurrent Neural Network (RNN) model."""
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1], 1)))
        model.add(SimpleRNN(50))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    def build_cnn_model(self):
        """
        Builds a CNN model.
        """
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1], 1)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    #def hyperparameter_tuning(self):
    #    """
    #    Performs hyperparameter tuning for classical models only.
    #    Skips neural networks (MLP, RNN, LSTM, CNN).
    #    """
    #    param_grids = {
    #        'Random Forest': {
    #            'classifier__n_estimators': [50, 100],
    #            'classifier__max_depth': [None, 5, 10]
    #        },
    #        'Gradient Boosting': {
    #            'classifier__learning_rate': [0.01, 0.1],
    #            'classifier__n_estimators': [50, 100]
    #        },
    #        'Logistic Regression': {
    #            'classifier__C': [0.1, 1, 10],
    #            'classifier__solver': ['liblinear', 'lbfgs']
    #        },
    #        'Decision Tree': {
    #            'classifier__max_depth': [None, 5, 10],
    #            'classifier__min_samples_split': [2, 5]
    #        }
    #    }
    #
    #    best_models = {}
    #    
    #    for name, model in self.models.items():
    #        # Skip neural networks (they don't have parameter grids)
    #        if name in ['MLP', 'RNN', 'LSTM', 'CNN']:
    #            continue  # Skip tuning for these models
    #        
    #        print(f"Tuning hyperparameters for {name}...")
    #        pipeline = Pipeline([
    #            ('scaler', StandardScaler()),
    #            ('classifier', model)
    #        ])
    #
    #        search = GridSearchCV(
    #            pipeline,
    #            param_grid=param_grids[name],
    #            cv=3,
    #            scoring='accuracy',
    #            n_jobs=-1
    #        )
    #        
    #        search.fit(self.X_train, self.y_train)
    #        best_models[name] = search.best_estimator_
    #        print(f"{name} best parameters: {search.best_params_}")
    #
    #    self.models.update(best_models)
    def hyperparameter_tuning(self):
        """Faster hyperparameter tuning with simplified grids."""
        param_grids = {
            'Random Forest': {'classifier__n_estimators': [100], 'classifier__max_depth': [5, 10]},
            'Gradient Boosting': {'classifier__learning_rate': [0.1], 'classifier__n_estimators': [100]},
            'Logistic Regression': {'classifier__C': [1, 10], 'classifier__solver': ['liblinear']},
            'Decision Tree': {'classifier__max_depth': [5, 10], 'classifier__min_samples_split': [2]}
        }
    
        best_models = {}
    
        for name, model in self.models.items():
            if name in ['MLP','RNN','LSTM','CNN']: 
                continue
            
            print(f"Quick tuning for {name}...")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
    
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grids[name],
                n_iter=2,
                cv=2,
                scoring='accuracy',
                n_jobs=-1
            )
    
            search.fit(self.X_train, self.y_train)
            best_models[name] = search.best_estimator_
    
        self.models.update(best_models)

    def train_and_evaluate(self):
        """
        Trains and evaluates all models, handling neural networks appropriately.
        """
        self.add_models()
        self.hyperparameter_tuning()
    
        best_model = None
        best_score = 0
        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                # Log model type and dataset info
                mlflow.log_param("dataset", "creditcard_or_fraud_data")  # Update name
                mlflow.log_param("model_type", name)

                start_time = time.time()
                
                # Handle all neural networks
                if name in ['MLP', 'RNN', 'LSTM', 'CNN']:
                    # MLP uses 2D data; others use 3D
                    if name == 'MLP':
                        X_train_reshaped = self.X_train.values.astype('float32')
                        X_test_reshaped = self.X_test.values.astype('float32')
                    else:
                        X_train_reshaped = self.X_train.values.reshape(
                            (self.X_train.shape[0], self.X_train.shape[1], 1)
                        ).astype('float32')
                        X_test_reshaped = self.X_test.values.reshape(
                            (self.X_test.shape[0], self.X_test.shape[1], 1)
                        ).astype('float32')
    
                    # Train neural network
                    model.fit(
                        X_train_reshaped, 
                        self.y_train, 
                        epochs=3, 
                        batch_size=64, 
                        verbose=0
                    )
                    
                    # Predict probabilities
                    y_prob = model.predict(X_test_reshaped).flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                    # Log Keras model
                    mlflow.keras.log_model(model, f"{name}_model")
                # Handle classical ML models
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_prob = model.predict_proba(self.X_test)[:, 1]  # Only for non-NN models
                    # Log sklearn model
                    mlflow.sklearn.log_model(model, f"{name}_model")   
                end_time = time.time()
                training_duration = end_time - start_time
                print(f"{name} took {training_duration:.2f} seconds to train")

                self.y_probs[name] = y_prob

                # Log metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_prob)

                # Log metrics
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc
                })

                # Log hyperparameters
                if name in ['Random Forest', 'Gradient Boosting']:
                    # Get the best parameters from the tuned model
                    best_params = self.models[name].get_params()
                    for param, value in best_params.items():
                        mlflow.log_param(param, value)

                # Save the model with a sanitized name
                model_name = name.replace(" ", "_").lower()

                # Log the model to MLflow
                if name in ['LSTM', 'CNN']:
                    mlflow.keras.log_model(model, f"{model_name}_model")
                else:
                    mlflow.sklearn.log_model(model, f"{model_name}_model")

                # Register the model in the MLflow Model Registry
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}_model"
                mlflow.register_model(model_uri, f"{model_name}")

                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = model
                    best_model_name = name

                self.performance_metrics[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'ROC AUC': roc_auc
                }
                print(f"{name} model trained and logged with MLflow")

        return best_model, best_model_name

    
    def save_best_models(self, best_model, best_model_name, dataset_name):
        """
        Saves the best model to disk for later use.
        """
        sanitized_name = best_model_name.replace(' ', '_').lower()
        joblib.dump(best_model, f"../app/{sanitized_name}_{dataset_name}_best_model.pkl")
        print(f"{best_model_name} best model saved.")

    def get_results(self):
        """
        Returns the evaluation results for all models.
        """
        return self.performance_metrics, self.y_probs