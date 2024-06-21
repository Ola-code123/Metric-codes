import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import time

# Add paths for the NSL-KDD train and test dataset
train_path = '...'
test_path = '...'

# Column names for the datasets
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# Load training and testing data without headers, with specified column names
train_data = pd.read_csv(train_path, header=None, names=col_names)
test_data = pd.read_csv(test_path, header=None, names=col_names)

# Label encoding for categorical columns 
cat_cols = ["protocol_type", "service", "flag"]
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Preparing feature and target datasets for training and testing
X_training = train_data.drop('label', axis=1)
y_training = train_data['label']
X_testing = test_data.drop('label', axis=1)
y_testing = test_data['label']

# Initialize and train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)


# Initialize SHAP Tree Explainer using the trained model
explainer = shap.TreeExplainer(model)

# Sample sizes
sample_sizes = [10, 100, 1000]

# Iterate over sample sizes and compute SHAP explaining time
for size in sample_sizes:
    if size <= len(X_testing):
        # Select a random sample from the test data
        sample_test = X_testing.sample(n=size, random_state=42)
        start_time = time.time()
        # Compute SHAP values for the sample
        _ = explainer.shap_values(sample_test)
        
        # Calculate and print the time taken to compute SHAP values
        shap_time = time.time() - start_time
        print(f"SHAP explanations for {size} samples computed in {shap_time:.2f} seconds")
    else:
        # Handling cases where the sample size exceeds available data
        print(f"Sample size {size} exceeds the number of available samples {len(X_testing)}.")
