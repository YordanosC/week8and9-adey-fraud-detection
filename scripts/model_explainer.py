import shap
import joblib
import pandas as pd
from lime import lime_tabular
import matplotlib.pyplot as plt
shap.initjs()

class ModelExplainer:
    """
    A class for explaining machine learning models using SHAP and LIME.

    Attributes:
    -----------
    model : object
        The trained machine learning model.
    X_test : DataFrame
        The test dataset to explain.

    Methods:
    ----------
    __init__(self, model_path, X_test):
        Initializes the class with the model and test dataset paths.
    
    explain_with_shap(self, instance_idx=0):
        Generates SHAP Summary Plot, Force Plot, and Dependence Plot for the given model.
    
    explain_with_lime(self, instance_idx=0):
        Generates LIME Feature Importance Plot for a single instance of the dataset.
    
    explain_model(self, instance_idx=0):
        Runs both SHAP and LIME explainability functions on the model and dataset.
    """

    def __init__(self, model_path, X_test):
        """
        Initialize the ModelExplainer class with the model and test dataset.

        Parameters:
        -----------
        model_path : str
            The path to the saved model file (e.g., .pkl).
        X_test : DataFrame
            The test dataset (in pandas DataFrame format).
        """
        self.model = joblib.load(model_path)  # Load the saved model
        self.X_test = X_test  # Load the test dataset

        # If the model is part of a scikit-learn pipeline, extract the last model from the pipeline
        if hasattr(self.model, 'steps'):
            self.model = self.model.steps[-1][1]  # Extract the model from the last step of the pipeline
    def _create_shap_explainer(self):
        """Auto-detect model type for SHAP and create the appropriate explainer."""
        model_type = type(self.model).__name__.lower()
        
        if 'tree' in model_type:  # Check if it's a tree-based model
            return shap.TreeExplainer(self.model)
        elif 'linear' in model_type:  # Check for linear models
            return shap.LinearExplainer(self.model, self.X_train)
        else:
            # KernelExplainer as a fallback for other types of models
            return shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 100))
    def explain_with_shap(self, instance_idx=0):
        """
        Generate SHAP Summary Plot, Force Plot, and Dependence Plot for the model.

        Parameters:
        -----------
        instance_idx : int, optional (default=0)
            The index of the instance to explain with SHAP Force Plot.
        """
        print("Generating SHAP explanations...")

        # Create SHAP explainer
        explainer_shap = shap.TreeExplainer(self.model)
        shap_values = explainer_shap.shap_values(self.X_test)
        print(shap_values)
        # Handle binary classification case (class 1 for fraud)
        if isinstance(shap_values, list):  # If shap_values is a list, it's for multiclass/binary classification
            shap_values_to_use = shap_values[1]  # Class 1 (Fraud) SHAP values
            base_value = explainer_shap.expected_value[1]  # Base value for class 1 (Fraud)
        else:
            shap_values_to_use = shap_values  # For models that only have one output (binary case)
            base_value = explainer_shap.expected_value  # Base value for binary case

        # Check if SHAP values and X_test have matching shapes
        print(f"Shape of SHAP values: {shap_values_to_use.shape}")
        print(f"Shape of X_test: {self.X_test.shape}")

        if shap_values_to_use.shape[0] != self.X_test.shape[0]:
            raise ValueError(f"Shape mismatch: SHAP values have {shap_values_to_use.shape[0]} rows, "
                            f"but X_test has {self.X_test.shape[0]} rows.")

        # SHAP Summary Plot: Overview of important features
        plt.figure(figsize=(15, 4))
        shap.summary_plot(shap_values_to_use, self.X_test, show=False)
        plt.title('SHAP Summary Plot')
        plt.show()

        # Sample only a small number of rows, such as the top 100 samples
        sample_indices = shap_values_to_use[:100]  # Sample the top 100 SHAP values

        # Choose a specific index from the subsample
        if instance_idx >= len(sample_indices):
            raise IndexError(f"Instance index {instance_idx} is out of bounds for the sampled data.")

        # Plot SHAP force plot for the selected instance from the subsampled data
        shap.plots.force(base_value, sample_indices[instance_idx])  # Base value and SHAP values for the selected instance
        plt.title(f'SHAP Force Plot for Sampled Instance {instance_idx}')
        plt.show()

        # SHAP Dependence Plot: Relationship between feature and model output
        shap.dependence_plot(self.X_test.columns[0], shap_values_to_use, self.X_test, show=False)
        plt.title(f'SHAP Dependence Plot for Feature: {self.X_test.columns[0]}')
        plt.show()



    def explain_with_lime(self, instance_idx=0):
        """
        Generate LIME Feature Importance Plot for a single instance of the dataset.

        Parameters:
        -----------
        instance_idx : int, optional (default=0)
            The index of the instance to explain with LIME.
        """
        print("Generating LIME explanations...")

        # Create LIME explainer
        explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=self.X_test.values, 
            feature_names=self.X_test.columns, 
            mode='classification'
        )

        # Select a single instance (default: first instance)
        instance = self.X_test.iloc[instance_idx].values
        explanation = explainer_lime.explain_instance(instance, self.model.predict_proba)

        # Display LIME Feature Importance Plot
        explanation.as_pyplot_figure()
        plt.title(f'LIME Feature Importance for Instance {instance_idx}')
        plt.show()

    def explain_model(self, instance_idx=0):
        """
        Run both SHAP and LIME explainability methods for the model.

        Parameters:
        -----------
        instance_idx : int, optional (default=0)
            The index of the instance to explain with LIME and SHAP.
        """
        # Explain the model with SHAP and LIME for the specified instance
        self.explain_with_shap(instance_idx)
        self.explain_with_lime(instance_idx)
