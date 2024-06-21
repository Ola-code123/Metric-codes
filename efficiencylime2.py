import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular
import time

# Paths for the datasets NSL-KDD
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

# Load the datasets
train_data = pd.read_csv(train_path, header=None, names=col_names)
test_data = pd.read_csv(test_path, header=None, names=col_names)

# Label encoding 
cat_cols = ["protocol_type", "service", "flag"]
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Drop columns
X_training = train_data.drop('label', axis=1)
y_training = train_data['label']
X_testing = test_data.drop('label', axis=1)
y_testing = test_data['label']

# Random Forest model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_training, y_training)

# Sample sizes 
sample_sizes = [10, 100, 1000]

# Starting LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_training.values,
    feature_names=X_training.columns.tolist(),
    class_names=['Normal', 'Attack'],
    mode='classification'
)

# Iterate over each sample size for LIME explanations
for size in sample_sizes:
    if size <= len(X_testing):
        sample_test = X_testing.sample(n=size, random_state=42)
        start_time = time.time()
        for i in range(size):
            exp = explainer.explain_instance(sample_test.iloc[i].values, model.predict_proba, num_features=3)
        lime_time = time.time() - start_time
        print(f"LIME explanation for all {size} samples computed in {lime_time:.2f} seconds")
    else:
        print(f"Sample size {size} exceeds the number of available samples {len(X_testing)}.")
