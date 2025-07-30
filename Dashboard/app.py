from flask import Flask, render_template, jsonify
import pandas as pd
import os
import logging

app = Flask(__name__)

LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs/api.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load fraud data from CSV
def load_fraud_data():
    try:
        logger.info('Loading fraud data from CSV...')
        data = pd.read_csv('../data/merged_fraud_data.csv')  # Adjusted path for simplicity
        logger.info('Fraud data loaded successfully.')
        return data
    except Exception as e:
        logger.error(f'Error loading fraud data: {e}')
        return None

def create_table_html(df, title=""):
    """
    Converts a Pandas DataFrame to an HTML table string.
    """
    table_html = f"<h2>{title}</h2>" if title else ""
    table_html += df.to_html(classes='data', index=False)  # Add a CSS class for styling
    return table_html

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to serve summary statistics as HTML
@app.route('/api/fraud_summary')
def fraud_summary():
    logger.info('Accessing /api/fraud_summary endpoint.')
    data = load_fraud_data()
    if data is None:
        return "Error: Data could not be loaded."

    total_transactions = len(data)
    total_fraud_cases = len(data[data['class'] == 1])
    fraud_percentage = (total_fraud_cases / total_transactions) * 100

    summary = {
        'Metric': ['Total Transactions', 'Total Fraud Cases', 'Fraud Percentage'],
        'Value': [total_transactions, total_fraud_cases, f"{fraud_percentage:.2f}%"]
    }

    df = pd.DataFrame(summary)
    table_html = create_table_html(df, "Fraud Summary")
    logger.info(f'Successfully generated fraud summary.')
    return table_html

# Endpoint to serve fraud trends as HTML
@app.route('/api/fraud_trends')
def fraud_trends():
    logger.info('Accessing /api/fraud_trends endpoint.')
    data = load_fraud_data()
    if data is None:
        return "Error: Data could not be loaded."

    data['purchase_date'] = pd.to_datetime(data['purchase_time']).dt.to_period('M').dt.strftime('%b %Y')
    trend_data = data.groupby('purchase_date').agg({
        'user_id': 'count',
        'class': lambda x: (x == 1).sum()
    }).reset_index()
    trend_data.rename(columns={'user_id': 'transaction_count', 'class': 'fraud_cases'}, inplace=True)
    trend_data = trend_data[['purchase_date', 'transaction_count', 'fraud_cases']]  # Reorder columns
    table_html = create_table_html(trend_data, "Fraud Trends")
    logger.info('Successfully generated fraud trends.')
    return table_html

# Endpoint to serve fraud by location as HTML
@app.route('/api/fraud_by_location')
def fraud_by_location():
    logger.info('Accessing /api/fraud_by_location endpoint.')
    data = load_fraud_data()
    if data is None:
        return "Error: Data could not be loaded."

    location_data = data.groupby('country').agg({
        'user_id': 'count',
        'class': lambda x: (x == 1).sum()
    }).reset_index()
    location_data.rename(columns={'user_id': 'transaction_count', 'class': 'fraud_cases'}, inplace=True)
    location_data = location_data[['country', 'transaction_count', 'fraud_cases']]  # Reorder columns
    table_html = create_table_html(location_data, "Fraud by Location")
    logger.info('Successfully generated fraud by location data.')
    return table_html

# Endpoint to serve fraud cases by top device and browser combinations as HTML
@app.route('/api/fraud_by_device_browser')
def fraud_by_device_browser():
    logger.info('Accessing /api/fraud_by_device_browser endpoint.')
    data = load_fraud_data()
    if data is None:
        return "Error: Data could not be loaded."

    fraud_data = data[data['class'] == 1]
    device_browser_data = (
        fraud_data.groupby(['device_id', 'browser'])
        .size()
        .reset_index(name='fraud_cases')
    )
    top_device_browser_data = device_browser_data.nlargest(10, 'fraud_cases')
    top_device_browser_data = top_device_browser_data[['device_id', 'browser', 'fraud_cases']]  # Reorder columns
    table_html = create_table_html(top_device_browser_data, "Top Fraud Cases by Device and Browser")
    logger.info('Successfully generated top fraud cases by device and browser data.')
    return table_html

if __name__ == '__main__':
    logger.info('Starting Flask application...')
    app.run(debug=True)