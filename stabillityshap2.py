import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
import shap
import matplotlib.pyplot as plt

# Add paths to the NSL-KDD train and test datasets
train_path = '...'
test_path = '...'

# Column names for the NSL-KDD datasets
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

# Load datasets
train_data = pd.read_csv(train_path, header=None, names=col_names)
test_data = pd.read_csv(test_path, header=None, names=col_names)

# Features for the model training
features_reduced = ['duration', 'service', 'flag', 'src_bytes', 'dst_bytes', 'hot', 'logged_in', 
                   'num_compromised', 'num_root', 'num_file_creations', 'count', 'srv_count', 
                   'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'dst_host_count', 
                   'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 
                   'dst_host_srv_serror_rate']

cat_cols = ["protocol_type", "service", "flag"]  

# Encoding with OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_data[cat_cols] = encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# Prepare datasets
X_training = train_data[features_reduced]
y_training = train_data['label']
X_testing = test_data[features_reduced]
y_testing = test_data['label']

# Train the RandomForest model with reduced features
model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
model.fit(X_training, y_training)

# Initialize SHAP explainer 
explainer = shap.TreeExplainer(model)


# Collecting top-k features across N runs
top_features_sets = []
for i in range(3):  
    np.random.seed(i)
    sample_indices = np.random.choice(X_testing.index, size=50, replace=False)  
    for idx in sample_indices:
        shap_values = explainer.shap_values(X_testing.iloc[idx:idx+1])
        feature_importance = np.abs(shap_values[1]).flatten()
        top_indices = np.argsort(feature_importance)[-10:]  
        top_features = set(X_testing.columns[top_indices])
        top_features_sets.append(top_features)  
        print(f"Run {i}, Instance {idx}, Top Features: {top_features}")


# Calculate stability 
if top_features_sets:  
    common_features = set.intersection(*top_features_sets)
    stability_score = len(common_features) / 10
    print(f"Stability Score: {stability_score}")
    print("Common features across runs:", ', '.join(common_features))
else:
    print("No common features")
