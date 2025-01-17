import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Set the directory where the files are located
# base_dir = r"C:\Users\ASUS\Downloads\Telecom Anomaly Detection"
# os.chdir(base_dir)  # Change to the specified directory

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
sns.histplot(df['feature'], kde=True)
plt.show()
# Standardizing the features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Isolation Forest ---
iso_forest = IsolationForest(contamination=0.1, random_state=42)  # contamination = proportion of outliers
y_pred_iso = iso_forest.fit_predict(X_scaled)
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
plot_file = 'anomaly_detection_isolation_forest.png'
plt.savefig(plot_file)
print(f"Anomaly detection plot (Isolation Forest) saved to {plot_file}.")
plt.show()

# --- Local Outlier Factor ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X_scaled)
data['Anomaly_LOF'] = y_pred_lof

# Visualize the anomalies detected by LOF
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['Latency'], 
    y=data['Signal_Strength'], 
    hue=data['Anomaly_LOF'], 
    palette={1: 'blue', -1: 'red'}
)
plt.title('Anomaly Detection using LOF')
plot_file = 'anomaly_detection_lof.png'
plt.savefig(plot_file)
print(f"Anomaly detection plot (LOF) saved to {plot_file}.")
plt.show()

# --- One-Class SVM ---
svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
y_pred_svm = svm.fit_predict(X_scaled)
data['Anomaly_SVM'] = y_pred_svm

# Visualize the anomalies detected by One-Class SVM
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['Energy_Efficiency'], 
    y=data['Signal_Strength'], 
    hue=data['Anomaly_SVM'], 
    palette={1: 'blue', -1: 'red'}
)
plt.title('Anomaly Detection using One-Class SVM')
plot_file = 'anomaly_detection_svm.png'
plt.savefig(plot_file)
print(f"Anomaly detection plot (One-Class SVM) saved to {plot_file}.")
plt.show()

# --- Pair Plots ---
pairplot_file = 'pairplot_isolation_forest.png'
sns.pairplot(data, hue="Anomaly_Isolation_Forest", vars=features, palette={1: 'blue', -1: 'red'})
plt.suptitle('Pair Plot for Isolation Forest Anomalies', y=1.02)
plt.savefig(pairplot_file)
print(f"Pair plot (Isolation Forest) saved to {pairplot_file}.")
plt.show()

pairplot_file = 'pairplot_lof.png'
sns.pairplot(data, hue="Anomaly_LOF", vars=features, palette={1: 'blue', -1: 'red'})
plt.suptitle('Pair Plot for LOF Anomalies', y=1.02)
plt.savefig(pairplot_file)
print(f"Pair plot (LOF) saved to {pairplot_file}.")
plt.show()

pairplot_file = 'pairplot_svm.png'
sns.pairplot(data, hue="Anomaly_SVM", vars=features, palette={1: 'blue', -1: 'red'})
plt.suptitle('Pair Plot for One-Class SVM Anomalies', y=1.02)
plt.savefig(pairplot_file)
print(f"Pair plot (One-Class SVM) saved to {pairplot_file}.")
plt.show()
