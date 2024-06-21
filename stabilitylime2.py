import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import time

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

# Defining features for the model
features_reduced = ['duration', 'service', 'flag', 'src_bytes', 'dst_bytes', 'hot', 'logged_in', 
                   'num_compromised', 'num_root', 'num_file_creations', 'count', 'srv_count', 
                   'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'dst_host_count', 
                   'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 
                   'dst_host_srv_serror_rate']

cat_cols = ["protocol_type", "service", "flag"]  

# Encoding with OrdinalEncoder
encoder = OrdinalEncoder()
train_data[cat_cols] = encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# Prepare datasets
X_training = train_data[features_reduced]
y_training = train_data['label']
X_testing = test_data[features_reduced]
y_testing = test_data['label']

# Training the RandomForest model
model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
model.fit(X_training, y_training)

# Initialize LIME explainer 
explainer = LimeTabularExplainer(
    training_data=X_training.values,
    feature_names=features_reduced,
    class_names=['Not Attack', 'Attack'],
    categorical_features=[X_training.columns.get_loc(c) for c in cat_cols if c in features_reduced],
    mode='classification'
)

# Function to extract feature names from explanations
def extract_feature_name(description):
    match = re.match(r"^(.*?)(\s*<=|<|=|>|>=)\s*[\d.+-]+", description)
    if match:
        return match.group(1)
    return None

# Stability analysis 
N = 3  # Number of runs
k = 10  # Top-k features
sample_size = 50  # Size of sample for each explanation run

# Collecting top-k features across N runs
top_features_sets = []
for i in range(N):
    sample = X_testing.sample(sample_size, random_state=i)
    indices = sample.index
    lime_features = []
    for idx in indices:
        exp = explainer.explain_instance(X_testing.loc[idx], model.predict_proba, num_features=k)
        features = set(extract_feature_name(feat[0]) for feat in exp.as_list() if extract_feature_name(feat[0]) is not None)
        lime_features.extend(features)
    top_features_sets.append(set(lime_features[:k]))

# Calculate stability
common_features = set.intersection(*top_features_sets)
stability_score = len(common_features) / k
print(f"Stability Score: {stability_score}")
print("Number of Common Features:", len(common_features))
print("Common Features:", ', '.join(common_features))
