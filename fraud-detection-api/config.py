import os

# Configuration settings
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'app/model/gradient_boosting_fraud_best_model.pkl')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'app/logs/api.log')

# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)