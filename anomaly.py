import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Set the directory where the files are located
base_dir = r"C:\Users\ASUS\Downloads\Telecom Anomaly Detection"
os.chdir(base_dir)  # Change to the specified directory

# Load dataset (assuming it's stored in a CSV file)
data_file = 'telecom_anomaly.csv'
data = pd.read_csv(data_file)

# Check for missing values and fill them if necessary
if data.isnull().sum().sum() > 0:
    print("Filling missing values with column means...")
    data.fillna(data.mean(), inplace=True)

# Features to be used for anomaly detection
features = ['Latency', 'Packet_Loss_Rate', 'Signal_Strength', 'Interference_Level', 'Energy_Efficiency']

# Extract the features for anomaly detection
X = data[features]

# Standardizing the features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
