**Telecom Anomaly Detection Documentation**

This documentation explains each step of the anomaly detection process, covering data preparation, processing, model training, and visualization.

---

### 1. **Importing Necessary Libraries**

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
```

- **os**: Used to navigate directories.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib** and **seaborn**: For data visualization.
- **scikit-learn**: For preprocessing and implementing anomaly detection algorithms.

---

### 2. **Setting the Working Directory**

```python
base_dir = r"C:\Users\ASUS\Downloads\Telecom Anomaly Detection"
os.chdir(base_dir)
```

- Changes the current working directory to the location where the data and scripts are stored.
- Ensures seamless loading and saving of files within the same directory.

---

### 3. **Loading the Dataset**

```python
data_file = 'telecom_anomaly.csv'
data = pd.read_csv(data_file)
```

- Reads the dataset `telecom_anomaly.csv` into a DataFrame for further processing.
- Assumes the file contains columns relevant to the problem domain, such as network parameters.

---

### 4. **Handling Missing Values**

```python
if data.isnull().sum().sum() > 0:
    print("Filling missing values with column means...")
    data.fillna(data.mean(), inplace=True)
```

- Checks for missing values in the dataset.
- If missing values are found, fills them with the mean of their respective columns to maintain consistency.
- Prints a message for transparency.

---

### 5. **Selecting Features for Anomaly Detection**

```python
features = ['Latency', 'Packet_Loss_Rate', 'Signal_Strength', 'Interference_Level', 'Energy_Efficiency']
X = data[features]
```

- Specifies the features (columns) relevant for detecting anomalies.
- Extracts these features into a new DataFrame `X` for preprocessing and model training.

---

### 6. **Standardizing the Features**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- Standardizes the features to have zero mean and unit variance.
- Standardization ensures better performance for distance-based algorithms like Isolation Forest and One-Class SVM.

---

### 7. **Anomaly Detection Using Isolation Forest**

```python
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_iso = iso_forest.fit_predict(X_scaled)
data['Anomaly_Isolation_Forest'] = y_pred_iso
```

- **Isolation Forest**:
  - A tree-based algorithm that isolates anomalies by randomly splitting data.
  - `contamination=0.1`: Assumes 10% of the data are anomalies.
  - `random_state=42`: Ensures reproducibility.
- Adds a new column `Anomaly_Isolation_Forest` to the DataFrame with anomaly labels (`1` for normal, `-1` for anomaly).

---

### 8. **Saving the Results**

```python
output_file = 'telecom_anomaly_with_labels.csv'
data.to_csv(output_file, index=False)
print(f"Processed data with anomaly labels saved to {output_file}.")
```

- Saves the updated DataFrame, including anomaly labels, to a new CSV file.
- Prints a confirmation message with the file name.

---

### 9. **Visualization for Isolation Forest**

```python
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
plot_file = 'anomaly_detection_plot.png'
plt.savefig(plot_file)
print(f"Anomaly detection plot saved to {plot_file}.")
plt.show()
```

- Creates a scatter plot of `Latency` vs. `Packet Loss Rate`, highlighting anomalies in red and normal points in blue.
- Saves the plot as `anomaly_detection_plot.png` for further analysis.

---

### 10. **Anomaly Detection Using Local Outlier Factor (LOF)**

```python
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X_scaled)
data['Anomaly_LOF'] = y_pred_lof
```

- **Local Outlier Factor**:
  - Identifies anomalies by measuring the local density deviation of data points.
  - `n_neighbors=20`: Considers 20 nearest neighbors.
  - `contamination=0.1`: Assumes 10% of the data are anomalies.
- Adds a new column `Anomaly_LOF` with anomaly labels.

---

### 11. **Visualization for LOF**

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Latency'], y=data['Signal_Strength'], hue=data['Anomaly_LOF'], palette={1: 'blue', -1: 'red'})
plt.title('Anomaly Detection using LOF')
plt.show()
```

- Visualizes anomalies detected by LOF using `Latency` and `Signal Strength` as axes.
- Highlights anomalies in red and normal points in blue.

---

### 12. **Anomaly Detection Using One-Class SVM**

```python
svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
y_pred_svm = svm.fit_predict(X_scaled)
data['Anomaly_SVM'] = y_pred_svm
```

- **One-Class SVM**:
  - A classification algorithm that separates normal data from outliers.
  - `nu=0.1`: Upper bound on the fraction of training errors (assumes 10% anomalies).
  - `kernel='rbf'`: Uses a radial basis function kernel.
- Adds a new column `Anomaly_SVM` with anomaly labels.

---

### 13. **Visualization for One-Class SVM**

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Energy_Efficiency'], y=data['Throughput'], hue=data['Anomaly_SVM'], palette={1: 'blue', -1: 'red'})
plt.title('Anomaly Detection using One-Class SVM')
plt.show()
```

- Visualizes anomalies detected by One-Class SVM using `Energy Efficiency` and `Throughput` as axes.

---

### 14. **Pair Plots for Comparison**

```python
sns.pairplot(data, hue="Anomaly_Isolation_Forest", vars=features, palette={1: 'blue', -1: 'red'})
plt.suptitle('Pair Plot for Isolation Forest Anomalies', y=1.02)
plt.show()

sns.pairplot(data, hue="Anomaly_LOF", vars=features, palette={1: 'blue', -1: 'red'})
plt.suptitle('Pair Plot for LOF Anomalies', y=1.02)
plt.show()

sns.pairplot(data, hue="Anomaly_SVM", vars=features, palette={1: 'blue', -1: 'red'})
plt.suptitle('Pair Plot for One-Class SVM Anomalies', y=1.02)
plt.show()
```

- Creates pair plots to compare the anomalies detected by each method across all features.
- Highlights anomalies in red and normal points in blue for visual analysis of relationships.

---

ok done

