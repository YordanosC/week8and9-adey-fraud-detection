import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import requests
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# Define Flask backend URL
FLASK_BACKEND_URL = "http://127.0.0.1:5000"  # Flask is running on port 5000

# Function to fetch data from Flask API
def fetch_data(endpoint):
    try:
        response = requests.get(f"{FLASK_BACKEND_URL}{endpoint}")
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {endpoint}: {e}")
        return None

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Fraud Insights Dashboard", style={'textAlign': 'center'}),

    html.Div([
        # Summary boxes
        html.Div([
            html.H3("Total Transactions"),
            html.Div(id='total-transactions')
        ], style={'width': '24%', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'margin': '5px'}),
        html.Div([
            html.H3("Total Fraud Cases"),
            html.Div(id='total-fraud-cases')
        ], style={'width': '24%', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'margin': '5px'}),
        html.Div([
            html.H3("Fraud Percentage"),
            html.Div(id='fraud-percentage')
        ], style={'width': '24%', 'display': 'inline-block', 'border': '1px solid black', 'padding': '10px', 'margin': '5px'}),
    ]),

    # Fraud Trends Line Chart
    html.H2("Fraud Trends Over Time"),
    dcc.Graph(id='fraud-trends-chart'),

    # Fraud by Location Map
    html.H2("Fraud by Location"),
    dcc.Graph(id='fraud-location-map'),

    # Fraud by Device and Browser Bar Chart
    html.H2("Fraud by Device and Browser"),
    dcc.Graph(id='fraud-device-browser-chart')
])

# Callback to update summary boxes
@app.callback(
    [Output('total-transactions', 'children'),
     Output('total-fraud-cases', 'children'),
     Output('fraud-percentage', 'children')],
    [Input('total-transactions', 'id')] # Just need an initial trigger, so using the ID itself.
)
def update_summary(dummy_input):
    summary_data = fetch_data('/api/fraud_summary')
    if summary_data:
        return (
            summary_data['total_transactions'],
            summary_data['total_fraud_cases'],
            f"{summary_data['fraud_percentage']:.2f}%"
        )
    else:
        return "Error", "Error", "Error"

# Callback to update fraud trends chart
@app.callback(
    Output('fraud-trends-chart', 'figure'),
    [Input('fraud-trends-chart', 'id')]
)
def update_fraud_trends_chart(dummy_input):
    trends_data = fetch_data('/api/fraud_trends')
    if trends_data:
        df = pd.DataFrame(trends_data)
        fig = px.line(df, x='purchase_date', y='fraud_cases', title='Fraud Cases Over Time')
        return fig
    else:
        return {}

# Callback to update fraud location map
@app.callback(
    Output('fraud-location-map', 'figure'),
    [Input('fraud-location-map', 'id')]
)
def update_fraud_location_map(dummy_input):
    location_data = fetch_data('/api/fraud_by_location')
    if location_data:
        df = pd.DataFrame(location_data)
        fig = px.choropleth(
            df,
            locations="country",  # Column with country names or ISO codes
            locationmode="country names",  # Use 'country names' or 'ISO-3' depending on your data
            color="fraud_cases",  # Column with fraud case counts
            hover_name="country",  # Column to display on hover
            color_continuous_scale=px.colors.sequential.Plasma,  # Choose a color scale
            title="Fraud Cases by Country"
        )
        return fig
    else:
        return {}

# Callback to update fraud device/browser chart
@app.callback(
    Output('fraud-device-browser-chart', 'figure'),
    [Input('fraud-device-browser-chart', 'id')]
)
def update_fraud_device_browser_chart(dummy_input):
    device_browser_data = fetch_data('/api/fraud_by_device_browser')
    if device_browser_data:
        df = pd.DataFrame(device_browser_data)
        fig = px.bar(df, x=['device_id', 'browser'], y='fraud_cases', title='Fraud Cases by Device and Browser')
        return fig
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)