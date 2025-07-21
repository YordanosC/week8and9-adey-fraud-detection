# Adey Fraud Detection
This repository implements a fraud detection pipeline for e-commerce transactions, using the CFPB dataset as a proxy for Fraud_Data.csv. The Interim-1 submission focuses on Task 1: data cleaning, EDA, feature engineering, and class imbalance strategy.
Project Structure
adey-fraud-detection/
├── data/
│   ├── raw/                   # Fraud_Data.csv, IpAddress_to_Country.csv
│   └── processed/             # Processed datasets
├── notebooks/
│   └── 1.0-data-analysis.ipynb
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

Setup Instructions

Clone: git clone https://github.com/yordanos/week8and9-adey-fraud-detection.git
Install dependencies: pip install -r requirements.txt
Run Docker: docker-compose up
Run notebook: Access via http://localhost:8888

Interim-1 Submission (July 20, 2025)

Task 1: Completed data cleaning, EDA, feature engineering (e.g., time_since_signup, IP-to-country mapping), and proposed SMOTE for class imbalance.

Requirements:

Place Fraud_Data.csv and IpAddress_to_Country.csv in data/raw/.
