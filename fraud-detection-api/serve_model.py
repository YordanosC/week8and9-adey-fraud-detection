from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from datetime import datetime
from config import MODEL_PATH, LOG_FILE

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for service health check"""
    logger.info("Health check requested")
    return jsonify({"status": "healthy", "timestamp": str(datetime.now())})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Log request
        logger.info(f"New prediction request from {request.remote_addr}")
        
        # Validate input
        data = request.get_json()
        if not data or 'features' not in data:
            logger.warning("Invalid request format")
            return jsonify({"error": "Invalid input format"}), 400
            
        # Convert features to numpy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1].tolist()
        
        # Log prediction
        logger.info(f"Prediction: {prediction[0]}, Probability: {probability[0]:.4f}")
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "timestamp": str(datetime.now())
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)