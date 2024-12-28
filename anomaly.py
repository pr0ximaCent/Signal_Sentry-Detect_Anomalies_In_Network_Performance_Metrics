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


# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)  # contamination = proportion of outliers
y_pred_iso = iso_forest.fit_predict(X_scaled)

# -1 indicates an anomaly, 1 indicates a normal observation
data['Anomaly_Isolation_Forest'] = y_pred_iso

# Save the DataFrame with anomaly labels to a new CSV file
output_file = 'telecom_anomaly_with_labels.csv'
data.to_csv(output_file, index=False)
print(f"Processed data with anomaly labels saved to {output_file}.")

# Visualize the anomalies detected by Isolation Forest
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['Latency'], 
    y=data['Packet_Loss_Rate'], 
    hue=data['Anomaly_Isolation_Forest'], 
    palette={1: 'blue', -1: 'red'}
)
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Latency')
plt.ylabel('Packet Loss Rate')

# Save the plot to a file
plot_file = 'anomaly_detection_plot.png'
plt.savefig(plot_file)
print(f"Anomaly detection plot saved to {plot_file}.")
plt.show()

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X_scaled)

# -1 indicates an anomaly, 1 indicates a normal observation
data['Anomaly_LOF'] = y_pred_lof

# Visualize the anomalies detected by LOF
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Latency'], y=data['Signal_Strength'], hue=data['Anomaly_LOF'], palette={1: 'blue', -1: 'red'})
plt.title('Anomaly Detection using LOF')
plt.show()